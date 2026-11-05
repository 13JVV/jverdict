"""합성데이터 생성기: 정답(효과·함정)을 알고 있는 패널 데이터를 만든다.

목적
- 툴이 심어 둔 효과를 제대로 복원하는지, 심어 둔 함정(측정 단절·결측·부분 집계·분류 변경·
  사전추세 위반)을 잡아내는지 정답과 대조해 정확도를 잰다.
- 실데이터에 없는 '효과 있음' 상황을 재현해 판정 엔진 전체를 시연한다.

구조 (두 층)
1. 실제 행동 층: 사람들이 실제로 탄 인원(또는 쓴 금액)
   log μ[g,t] = log 기준값[g] + 추세 + 요일 + 연중 계절 + 방학 + 명절 + 공통 잡음[t] + 집단 잡음[g,t]
                + 정책 효과 + 교란 충격 + 사전추세
   값 ~ Poisson(μ). 집계 단위(정류소·업종)가 있으면 단위별 몫으로 나눈다.
2. 측정 층: 데이터에 기록된 값
   측정 단절(일부만 집계, 다른 분류로 이동), 전체·집단 결측, 부분 집계일, 전체 집계 이상.

같은 시드면 언제나 같은 데이터가 나온다(재현성). 정답은 `SynthData.truth`에 모두 남는다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# 실제 설·추석(양력). 명절 연휴 3일 동안 이동이 줄어드는 효과를 넣는다.
LUNAR_HOLIDAYS = [
    "2018-02-16", "2018-09-24", "2019-02-05", "2019-09-13", "2020-01-25", "2020-10-01",
    "2021-02-12", "2021-09-21", "2022-02-01", "2022-09-10", "2023-01-22", "2023-09-29",
    "2024-02-10", "2024-09-17", "2025-01-29", "2025-10-06", "2026-02-17", "2026-09-25",
]


# --------------------------------------------------------------------------- 사양
@dataclass
class GroupSpec:
    name: str
    base: float                      # 기간당 평균 값 (일 단위면 하루 인원)
    vacation: float = 0.0            # 방학 중 로그 효과 (예: 청소년 −0.35)
    weekend: float = -0.2            # 주말 로그 효과 (일 단위만)
    noise_sd: float = 0.07           # 집단 고유 잡음의 정상 상태 표준편차
    trend: float = 0.0               # 연간 로그 추세


@dataclass
class Effect:
    """실제 정책 효과. size는 로그 크기(0.2 ≈ +22%)."""
    group: str
    start: str
    size: float
    ramp: int = 0                    # 효과가 다 나타나기까지 걸리는 기간 (0 = 즉시)
    end: str | None = None           # 일시적 효과면 끝나는 시점
    kind: str = "effect"


@dataclass
class PreTrend:
    """정책 전부터 대상 집단만 따로 움직이는 추세 (평행추세 위반)."""
    group: str
    slope: float                     # 연간 로그 기울기
    start: str
    end: str
    kind: str = "pretrend"


@dataclass
class Shock:
    """정책과 같은 때 일부 집단에만 생긴 다른 사건 (교란)."""
    groups: list[str]
    start: str
    end: str
    size: float
    kind: str = "shock"


@dataclass
class MeasurementBreak:
    """집계 방식 변경: start부터 대상 값의 keep 비율만 기록된다.

    adoption 기간에 걸쳐 서서히 빠진다(카드 등록이 퍼지는 모양).
    reclass_to가 있으면 빠진 값의 reclass_share만큼 그 분류로 옮겨 기록한다(새 분류도 가능).
    """
    group: str
    start: str
    keep: float
    adoption: int = 0
    reclass_to: str | None = None
    reclass_share: float = 1.0
    kind: str = "measurement_break"


@dataclass
class Outage:
    """모든 집단이 함께 덜 집계되는 기간 (시스템 이상)."""
    start: str
    end: str
    keep: float
    kind: str = "outage"


@dataclass
class Missing:
    """행이 아예 없는 기간. groups가 None이면 전체 결측."""
    start: str
    end: str
    groups: list[str] | None = None
    partial_tail: int = 0            # 전체 결측 뒤 부분 집계로 재개되는 기간 수
    kind: str = "missing"


@dataclass
class PartialDays:
    """흩어진 부분 집계일 n개 (모든 집단이 share만큼만 기록)."""
    n: int
    share_low: float = 0.02
    share_high: float = 0.30
    kind: str = "partial_days"


Injection = Effect | PreTrend | Shock | MeasurementBreak | Outage | Missing | PartialDays


@dataclass
class SynthSpec:
    start: str
    end: str
    groups: list[GroupSpec]
    freq: Literal["D", "MS"] = "D"
    units: int = 1                   # 집단별 집계 단위 수 (정류소·업종)
    unit_sd: float = 0.6             # 단위 규모의 로그 표준편차
    unit_noise: float = 0.05         # 단위별 기간 잡음
    common_sd: float = 0.05          # 모든 집단 공통 잡음 표준편차
    ar_phi: float = 0.45             # 잡음 자기상관 (일 단위 기준)
    seasonal_amp: float = 0.06       # 연중 계절 진폭
    holidays: bool = True
    injections: list = field(default_factory=list)
    drop_zero: bool = True           # 값이 0인 행은 기록하지 않음 (실데이터와 같은 방식)


@dataclass
class SynthData:
    df: pd.DataFrame                 # 긴 형식: time, group, unit, value
    truth: dict
    latent: pd.DataFrame             # 실제 행동 층의 집단 합계 (측정 전)
    spec: SynthSpec
    seed: int


# --------------------------------------------------------------------------- 도우미
def _index(spec: SynthSpec) -> pd.DatetimeIndex:
    return pd.date_range(spec.start, spec.end, freq=spec.freq)


def _ar1(rng: np.random.Generator, n: int, sd: float, phi: float) -> np.ndarray:
    """정상 상태 표준편차가 sd인 AR(1) 잡음."""
    if sd <= 0:
        return np.zeros(n)
    innov = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + innov[i]
    return x


def _vacation_mask(idx: pd.DatetimeIndex, rng: np.random.Generator, freq: str) -> np.ndarray:
    """학교 방학. 해마다 시작·끝이 며칠씩 달라진다."""
    if freq == "MS":
        return np.isin(idx.month, [1, 2, 8])
    m = np.zeros(len(idx), dtype=bool)
    for y in range(idx[0].year - 1, idx[-1].year + 1):
        segs = [(pd.Timestamp(f"{y}-07-20"), pd.Timestamp(f"{y}-08-20")),
                (pd.Timestamp(f"{y}-12-28"), pd.Timestamp(f"{y + 1}-02-25"))]
        for s, e in segs:
            s += pd.Timedelta(days=int(rng.integers(-5, 6)))
            e += pd.Timedelta(days=int(rng.integers(-5, 6)))
            m |= (idx >= s) & (idx <= e)
    return m


def _holiday_effect(idx: pd.DatetimeIndex, freq: str) -> np.ndarray:
    eff = np.zeros(len(idx))
    if freq != "D":
        return eff
    for h in LUNAR_HOLIDAYS:
        c = pd.Timestamp(h)
        for k, v in ((-1, -0.25), (0, -0.45), (1, -0.30)):
            eff[idx == c + pd.Timedelta(days=k)] += v
    return eff


def _pos(idx: pd.DatetimeIndex, date: str) -> int:
    return int(idx.searchsorted(pd.Timestamp(date)))


def _progress(n: int, i0: int, ramp: int) -> np.ndarray:
    """정책 시점부터 0 → 1로 가는 진행률. ramp가 0이면 계단."""
    p = np.zeros(n)
    if i0 >= n:
        return p
    if ramp <= 0:
        p[i0:] = 1.0
    else:
        k = np.arange(n - i0)
        p[i0:] = np.clip((k + 1) / ramp, 0, 1)
    return p


def _adoption(n: int, i0: int, days: int) -> np.ndarray:
    """로지스틱 모양의 확산(측정 단절이 퍼지는 모양)."""
    p = np.zeros(n)
    if i0 >= n:
        return p
    if days <= 0:
        p[i0:] = 1.0
        return p
    k = np.arange(n - i0)
    p[i0:] = 1 / (1 + np.exp(-(k - days / 2) / max(days / 8, 0.5)))
    return p


# --------------------------------------------------------------------------- 생성
def generate(spec: SynthSpec, seed: int = 0) -> SynthData:
    rng = np.random.default_rng(seed)
    idx = _index(spec)
    n = len(idx)
    daily = spec.freq == "D"
    years = (idx - idx[0]).days.to_numpy() / 365.25
    phi = spec.ar_phi if daily else spec.ar_phi ** 0.5
    vac = _vacation_mask(idx, rng, spec.freq)
    hol = _holiday_effect(idx, spec.freq) if spec.holidays else np.zeros(n)
    doy = idx.day_of_year.to_numpy()
    season = spec.seasonal_amp * np.sin(2 * np.pi * (doy - 100) / 365.25)
    common = _ar1(rng, n, spec.common_sd, phi)
    weekend = (idx.day_of_week >= 5) if daily else np.zeros(n, dtype=bool)

    names = [g.name for g in spec.groups]
    logmu = {}
    for g in spec.groups:
        lm = (np.log(g.base) + g.trend * years + season + g.vacation * vac + hol + common
              + g.weekend * weekend + _ar1(rng, n, g.noise_sd, phi))
        logmu[g.name] = lm

    truth = {"seed": seed, "freq": spec.freq, "period": [str(idx[0].date()), str(idx[-1].date())],
             "groups": names, "injections": []}

    # 1) 실제 행동 층
    for inj in spec.injections:
        if isinstance(inj, Effect):
            i0 = _pos(idx, inj.start)
            p = _progress(n, i0, inj.ramp)
            if inj.end:
                p[_pos(idx, inj.end) + 1:] = 0.0
            logmu[inj.group] = logmu[inj.group] + inj.size * p
        elif isinstance(inj, PreTrend):
            a, b = _pos(idx, inj.start), _pos(idx, inj.end)
            ramp = np.zeros(n)
            span = (idx[a:b + 1] - idx[a]).days.to_numpy() / 365.25
            ramp[a:b + 1] = span
            ramp[b + 1:] = span[-1] if len(span) else 0
            logmu[inj.group] = logmu[inj.group] + inj.slope * ramp
        elif isinstance(inj, Shock):
            a, b = _pos(idx, inj.start), _pos(idx, inj.end)
            for gname in inj.groups:
                logmu[gname][a:b + 1] += inj.size
        else:
            continue          # 측정 층 주입은 아래에서 기록한다
        truth["injections"].append(asdict(inj))

    # 집계 단위로 나누고 실제 값을 뽑는다
    unit_names = [f"U{u:03d}" for u in range(spec.units)]
    shares = {}
    for gname in names:
        w = np.exp(rng.normal(0, spec.unit_sd, spec.units))
        shares[gname] = w / w.sum()
    true_counts = {}
    for gname in names:
        mu = np.exp(logmu[gname])[:, None] * shares[gname][None, :]
        if spec.units > 1 and spec.unit_noise > 0:
            mu = mu * np.exp(rng.normal(0, spec.unit_noise, mu.shape))
        true_counts[gname] = rng.poisson(mu).astype(np.int64)
    latent = pd.DataFrame({g: true_counts[g].sum(axis=1) for g in names}, index=idx)

    # 2) 측정 층
    obs = {g: c.copy() for g, c in true_counts.items()}
    for inj in spec.injections:
        if isinstance(inj, MeasurementBreak):
            i0 = _pos(idx, inj.start)
            loss = (1 - inj.keep) * _adoption(n, i0, inj.adoption)
            keep_p = np.clip(1 - loss, 0, 1)[:, None]
            kept = rng.binomial(obs[inj.group], np.broadcast_to(keep_p, obs[inj.group].shape))
            moved = obs[inj.group] - kept
            obs[inj.group] = kept
            if inj.reclass_to:
                add = rng.binomial(moved, inj.reclass_share)
                if inj.reclass_to not in obs:
                    obs[inj.reclass_to] = np.zeros_like(add)
                obs[inj.reclass_to] = obs[inj.reclass_to] + add
        elif isinstance(inj, Outage):
            a, b = _pos(idx, inj.start), _pos(idx, inj.end)
            for g in obs:
                obs[g][a:b + 1] = rng.binomial(obs[g][a:b + 1], inj.keep)
    partial_truth = []
    for inj in spec.injections:
        if isinstance(inj, PartialDays):
            cand = np.arange(n)
            picked = np.sort(rng.choice(cand, size=min(inj.n, n), replace=False))
            for i in picked:
                share = rng.uniform(inj.share_low, inj.share_high)
                for g in obs:
                    obs[g][i] = rng.binomial(obs[g][i], share)
                partial_truth.append(str(idx[i].date()))
            truth["injections"].append({**asdict(inj), "dates": partial_truth})
        elif isinstance(inj, (MeasurementBreak, Outage)):
            truth["injections"].append(asdict(inj))

    # 긴 형식으로 만들고 결측을 적용한다
    frames = []
    for g, arr in obs.items():
        f = pd.DataFrame(arr, index=idx, columns=unit_names).stack().rename("value").reset_index()
        f.columns = ["time", "unit", "value"]
        f["group"] = g
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)[["time", "group", "unit", "value"]]
    if spec.drop_zero:
        df = df[df["value"] > 0]
    # 새로 생긴 분류는 처음 값이 생긴 때부터만 존재한다 (drop_zero=False여도)
    for g in set(obs) - set(names):
        first = df.loc[(df.group == g) & (df.value > 0), "time"].min()
        df = df[(df.group != g) | (df.time >= first)]

    for inj in spec.injections:
        if isinstance(inj, Missing):
            s, e = pd.Timestamp(inj.start), pd.Timestamp(inj.end)
            sel = (df.time >= s) & (df.time <= e)
            if inj.groups:
                sel &= df.group.isin(inj.groups)
            df = df[~sel]
            rec = asdict(inj)
            if inj.partial_tail and not inj.groups:
                tail = pd.date_range(e, periods=inj.partial_tail + 1, freq=spec.freq)[1:]
                fr = np.linspace(0.01, 0.25, len(tail))
                for t, sh in zip(tail, fr):
                    m = df.time == t
                    df.loc[m, "value"] = rng.binomial(df.loc[m, "value"].to_numpy(), sh)
                rec["tail_dates"] = [str(t.date()) for t in tail]
                if spec.drop_zero:
                    df = df[df["value"] > 0]
            truth["injections"].append(rec)

    df = df.sort_values(["time", "group", "unit"]).reset_index(drop=True)
    return SynthData(df=df, truth=truth, latent=latent, spec=spec, seed=seed)
