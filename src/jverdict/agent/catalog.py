"""데이터 카탈로그 (catalog.search, data.query의 기반).

Agent는 원자료 행을 직접 보지 않는다. 등록된 데이터셋을 이름으로 고르고,
'집단 × 단위 묶음'으로 분석할 계열을 정의하면 이 모듈이 집계 결과만 만들어 준다.

계열 정의 예
    {"groups": ["청소년"]}                              버스: 이용자 유형 '청소년'의 모든 정류소 합계
    {"groups": ["제주도민"], "units": "tier:가능"}       카드: 도민의 지원금 사용 가능 업종 합계
    {"groups": ["제주도민"], "units": ["면세점"]}         카드: 특정 업종만
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from jverdict.config import PROCESSED_DIR, RAW_DIR, ROOT
from jverdict.data import sources
from jverdict.quality._panel import to_wide

SECTOR_TIERS = ROOT / "src" / "jverdict" / "cases" / "relief2020_sectors.csv"


@dataclass
class Dataset:
    id: str
    title: str                 # 허브 데이터명
    source: str                # 원본 파일
    freq: str                  # D / MS
    group_label: str           # 집단 열의 뜻 (이용자 유형 등)
    unit_label: str            # 단위 열의 뜻 (정류소, 업종 등)
    value_label: str
    notes: list[str] = field(default_factory=list)
    synthetic: bool = False

    def card(self) -> dict:
        d = asdict(self)
        d.pop("synthetic")
        return d


DATASETS: dict[str, Dataset] = {d.id: d for d in [
    Dataset("bus_station", "이용자 유형별 버스정류소 이용 인원 현황", "이용자유형별버스정류소이용인원현황.csv", "D",
            "이용자 유형", "정류소", "이용 인원(명)",
            ["2019-01-01 ~ 2025-10-25, 일 단위", "이용자 유형 9종(일반·청소년·어린이·경로·장애·유공·기타)"]),
    Dataset("bus_route", "일별 버스 승차 인원 수 현황", "일별버스승차인원수현황.csv", "D",
            "이용자 유형", "노선", "승차 인원(명)", ["2019-01-01 ~ 2025-10-19, 일 단위, 승차 기준"]),
    Dataset("card_sector", "업종별 성별 카드 이용 정보", "업종별성별카드이용정보.csv", "MS",
            "이용자 구분", "업종", "이용 금액(원)",
            ["2018-01 ~ 2025-06, 월 단위, 비씨카드 실적", "이용자 구분: 제주도민·내국인·법인·중국·일본·동남아·기타외국",
             "업종 묶음 tier:가능 / tier:불가 / tier:불확실 (1차 재난지원금 사용처 규정 기준)"]),
    Dataset("synthetic_bus", "합성 버스 데이터 (정답을 아는 시연용)", "jverdict.synth 생성기", "D",
            "이용자 유형", "-", "이용 인원(명)",
            ["2021-01-01 ~ 2025-10-31, 일 단위", "실데이터 변동에 맞춘 합성데이터. 정답(심어 둔 효과)은 판정 후에만 공개"],
            synthetic=True),
    Dataset("synthetic_bus_event", "합성 버스 데이터 2 (같은 시기 행사가 섞인 시연용)", "jverdict.synth 생성기", "D",
            "이용자 유형", "-", "이용 인원(명)",
            ["2021-01-01 ~ 2025-10-31, 일 단위", "정책 효과는 없고, 정책 2주 뒤부터 10주 동안 청소년 대상 행사가 이용을 늘린 합성데이터. "
             "정답은 판정 후에만 공개"], synthetic=True),
]}


# --------------------------------------------------------------------------- 적재 (긴 형식: time, group, unit, value)
@lru_cache(maxsize=4)
def _load_bus(key: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{key}.parquet"
    if key == "bus_route_usertype_daily":
        q = f"select base_date time, user_type grp, bus_number unit, sum(user_count) value from '{path}' where act_type = '승차' group by all"
    else:
        q = f"select base_date time, user_type grp, station_name unit, sum(user_count) value from '{path}' group by all"
    df = duckdb.sql(q).df().rename(columns={"grp": "group"})
    df["time"] = pd.to_datetime(df["time"])
    return df


@lru_cache(maxsize=1)
def _load_card() -> pd.DataFrame:
    df = duckdb.execute(
        """
        SELECT strptime(ym || '01', '%Y%m%d')::DATE AS time, ut AS grp, replace(biz, '?', '') AS unit,
               sum(amt)::DOUBLE AS value
        FROM read_csv(?, header=false, skip=1, delim=',',
             columns={'ym':'VARCHAR','biz':'VARCHAR','ut':'VARCHAR','sex':'VARCHAR',
                      'users':'BIGINT','cnt':'BIGINT','amt':'BIGINT'})
        GROUP BY ALL
        """, [str(RAW_DIR / DATASETS["card_sector"].source)]).df().rename(columns={"grp": "group"})
    df["time"] = pd.to_datetime(df["time"])
    return df


_SYNTH: dict[int, tuple] = {}


def synthetic_truth(seed: int, dataset_id: str = "synthetic_bus") -> dict:
    return (_synthetic_event(seed) if dataset_id == "synthetic_bus_event" else _synthetic(seed))[1]


EVENT_F = {"policy": "2024-09-02", "start": "2024-09-16", "end": "2024-11-24", "size": 0.40}


def _synthetic_event(seed: int):
    """시연 사례 F: 2024-09-02 정책의 실제 효과는 0. 2024-09-16~11-24 청소년 대상 행사로 이용이 +49%(로그 0.40) 늘었다.
    판정 규칙만으로는 행사 효과를 정책 효과로 읽기 쉽고, Skeptic이 사건 달력의 행사 기간을 빼고 다시 추정하면 사라진다."""
    key = ("F", seed)
    if key not in _SYNTH:
        from jverdict.synth.generator import Effect, PartialDays, SynthSpec, generate
        from jverdict.synth.scenarios import BUS_GROUPS
        e = EVENT_F
        spec = SynthSpec("2021-01-01", "2025-10-31", BUS_GROUPS, "D",
                         injections=[Effect("청소년", e["start"], e["size"], end=e["end"]), PartialDays(6, share_high=0.25)])
        sd = generate(spec, seed)
        truth = {"effect_log": 0.0, "effect_pct": 0.0, "date": e["policy"], "group": "청소년", "seed": seed,
                 "note": f"정책 효과는 0. {e['start']}~{e['end']} 행사로 청소년 이용 +{np.expm1(e['size']) * 100:.0f}%(정책과 무관)"}
        _SYNTH[key] = (sd.df.copy(), truth)
    return _SYNTH[key]


def _synthetic(seed: int):
    """시연용 합성 데이터: 2024-03-04에 청소년 +22%(로그 0.20) 실제 효과를 심어 둔다."""
    if seed not in _SYNTH:
        from jverdict.synth.generator import Effect, PartialDays, SynthSpec, generate
        from jverdict.synth.scenarios import BUS_GROUPS
        spec = SynthSpec("2021-01-01", "2025-10-31", BUS_GROUPS, "D",
                         injections=[Effect("청소년", "2024-03-04", 0.20), PartialDays(6, share_high=0.25)])
        sd = generate(spec, seed)
        truth = {"effect_log": 0.20, "effect_pct": float(np.expm1(0.20)), "date": "2024-03-04", "group": "청소년",
                 "seed": seed, "partial_days": next(i for i in sd.truth["injections"] if i["kind"] == "partial_days")["dates"]}
        _SYNTH[seed] = (sd.df.copy(), truth)
    return _SYNTH[seed]


_FILES = {"bus_station": lambda: PROCESSED_DIR / "bus_station_usertype_daily.parquet",
          "bus_route": lambda: PROCESSED_DIR / "bus_route_usertype_daily.parquet",
          "card_sector": lambda: RAW_DIR / DATASETS["card_sector"].source}


def available(dataset_id: str) -> bool:
    """이 PC에 분석할 자료가 있는가. 합성데이터는 늘 있음. 공개 저장소·웹 데모에는 허브 자료를 싣지 않는다."""
    if dataset_id not in DATASETS:
        return False
    f = _FILES.get(dataset_id)
    return True if f is None else f().exists()


def load(dataset_id: str, seed: int = 20261104) -> pd.DataFrame:
    if dataset_id not in DATASETS:
        raise KeyError(f"등록되지 않은 데이터셋: {dataset_id}. 가능: {list(DATASETS)}")
    if dataset_id == "bus_station":
        return _load_bus("bus_station_usertype_daily")
    if dataset_id == "bus_route":
        return _load_bus("bus_route_usertype_daily")
    if dataset_id == "card_sector":
        return _load_card()
    if dataset_id == "synthetic_bus_event":
        return _synthetic_event(seed)[0]
    return _synthetic(seed)[0]


def fingerprint(dataset_id: str) -> dict:
    """재실행 기록용 데이터 지문(원본 파일 기준). 합성데이터는 생성 코드와 시드가 지문이다.

    원본 CSV가 없는 PC(배포 묶음에는 1GB 버스 원본을 싣지 않음)에서는 Agent가 실제로 읽는 가공 파일의 지문을
    원본이 있던 PC에서 적어 둔 기록(data/processed/_sources.json)과 맞춰 보고, 맞을 때만 기록된 원본 지문을 쓴다.
    그래서 어느 PC에서든 같은 데이터면 같은 지문이 나온다.
    """
    ds = DATASETS[dataset_id]
    if ds.synthetic:
        return {"dataset": dataset_id, "source": ds.source}
    p = RAW_DIR / ds.source
    if p.exists():
        raw = {"bytes": p.stat().st_size, "sha256": sources.sha256_file(p)}
    else:
        raw = sources.raw_fingerprint_via_processed(ds.source)
    return {"dataset": dataset_id, "source": ds.source, "bytes": raw["bytes"], "sha256": raw["sha256"]}


# --------------------------------------------------------------------------- 계열 만들기
@lru_cache(maxsize=1)
def sector_tiers() -> dict[str, list[str]]:
    t = pd.read_csv(SECTOR_TIERS)
    return {k: g.biz.tolist() for k, g in t.groupby("tier")}


def _units(spec_units, dataset_id: str) -> list[str] | None:
    if spec_units is None:
        return None
    if isinstance(spec_units, str):
        if spec_units.startswith("tier:"):
            parts = spec_units[5:].split("+")
            tiers = sector_tiers()
            out = []
            for p in parts:
                if p not in tiers:
                    raise KeyError(f"없는 업종 묶음: {p}. 가능: {list(tiers)}")
                out += tiers[p]
            return out
        return [spec_units]
    return list(spec_units)


def series_frame(dataset_id: str, series: dict[str, dict], seed: int = 20261104) -> tuple[pd.DataFrame, pd.DataFrame]:
    """계열 정의 → (긴 형식 집계: time, group=계열 이름, unit, value), 넓은 형식(time × 계열)."""
    df = load(dataset_id, seed)
    ds = DATASETS[dataset_id]
    parts = []
    for name, spec in series.items():
        groups = spec.get("groups")
        sel = df["group"].isin(groups) if groups else np.ones(len(df), bool)
        units = _units(spec.get("units"), dataset_id)
        if units is not None:
            sel &= df["unit"].isin(units)
        sub = df[sel]
        if sub.empty:
            raise ValueError(f"계열 '{name}'에 해당하는 행이 없음: {spec}")
        parts.append(sub.groupby(["time", "unit"], as_index=False)["value"].sum().assign(group=name))
    long = pd.concat(parts, ignore_index=True)[["time", "group", "unit", "value"]]
    wide = to_wide(long, "time", "group", "value", ds.freq)
    return long, wide


def profile(dataset_id: str, before: str | None = None, seed: int = 20261104) -> dict:
    """데이터셋 요약(집계만). before를 주면 그 이전 기간만 본다(설계 단계의 '효과 가림')."""
    df = load(dataset_id, seed)
    if before:
        df = df[df.time < pd.Timestamp(before)]
    g = df.groupby("group").agg(first=("time", "min"), last=("time", "max"), periods=("time", "nunique"),
                                units=("unit", "nunique"), total=("value", "sum"))
    per = df.groupby(["group", "time"]).value.sum().groupby("group").mean()
    g["mean_per_period"] = per
    g = g.sort_values("total", ascending=False)
    out = {"dataset": DATASETS[dataset_id].card(), "blinded_before": before,
           "period": [str(df.time.min().date()), str(df.time.max().date())],
           "groups": [{"group": k, "first": str(r["first"].date()), "last": str(r["last"].date()),
                       "periods": int(r.periods), "units": int(r.units),
                       "mean_per_period": round(float(r.mean_per_period), 1)} for k, r in g.iterrows()]}
    if dataset_id == "card_sector":
        out["unit_sets"] = {f"tier:{k}": len(v) for k, v in sector_tiers().items()}
    return out


def card_panel(groups: list[str], months: list[str], base_year: int = 2019, complete_only: bool = True) -> pd.DataFrame:
    """카드 패널: 단위 = 이용자 구분 × 업종, y = log(금액 / 기준 연도 같은 달 금액).

    사례 D 사전 등록(4장)의 결과 변수 정의를 일반화한 것이다. complete_only면 분석 월이 모두 있는 단위만 남긴다.
    """
    df = load("card_sector")
    df = df[df.group.isin(groups)]
    months_ts = pd.DatetimeIndex(pd.to_datetime(months))
    base = (df[df.time.dt.year == base_year].assign(cal=lambda d: d.time.dt.month)
            .rename(columns={"value": "base"})[["group", "unit", "cal", "base"]])
    win = df[df.time.isin(months_ts)].assign(cal=lambda d: d.time.dt.month)
    p = win.merge(base, on=["group", "unit", "cal"], how="inner")
    p = p[(p.value > 0) & (p.base > 0)].copy()
    p["y"] = np.log(p.value / p.base)
    if complete_only:
        full = p.groupby(["group", "unit"]).time.nunique().loc[lambda s: s == len(months_ts)].index
        p = p.set_index(["group", "unit"]).loc[full].reset_index()
    p = p.rename(columns={"unit": "biz"})
    p["unit"] = p.group + "|" + p.biz
    p["sm"] = p.biz + "|" + p.time.dt.strftime("%Y-%m")
    w = df[df.time.dt.year == base_year].groupby(["group", "unit"]).value.sum().rename("w_base").reset_index()
    p = p.merge(w.rename(columns={"unit": "biz"}), on=["group", "biz"], how="left")
    return p[["group", "biz", "unit", "time", "sm", "y", "w_base"]].reset_index(drop=True)
