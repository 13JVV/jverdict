"""검증 툴: 추정 결과를 반박해 본다.

- placebo_time  : 가짜 시행일마다 같은 추정을 반복해 '정책이 없을 때의 효과 분포'를 만든다.
                  실제 효과의 순위 p값과, 그 분포로 만든 경험적 신뢰구간을 낸다.
                  가짜 창이 실제 시행일(과 계절 차분 메아리)을 포함하면 뺀다.
- placebo_space : 정책 대상이 아닌 집단을 가짜 처치로 두고 같은 추정을 한다.
- sensitivity   : 기간·대조 집단·가중치 등을 바꿔도 결론이 유지되는지 본다.
- double_compute: 핵심 수치를 서로 다른 두 경로(pandas, SQL)로 계산해 대조한다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import duckdb
import numpy as np
import pandas as pd

from .estimate import Estimate


@dataclass
class PlaceboTime:
    real: float
    effects: list[float]
    dates: list[str]
    p_two: float               # 양측 순위 p: (1 + #|가짜| ≥ |실제|) / (1 + n)
    p_dir: float               # 실제 효과 방향의 단측 순위 p
    sd: float
    ci: tuple[float, float] | None   # 경험적 신뢰구간: 실제 − 가짜 분포 분위수
    n: int
    step: int
    note: str = ""

    @property
    def available(self) -> bool:
        return self.n >= 19            # 순위 p가 0.05 아래로 내려갈 수 있는 최소 개수

    def to_dict(self) -> dict:
        d = asdict(self)
        d["available"] = self.available
        return d


def placebo_time(fit: Callable[[int], Estimate | None], n_index: int, i0: int, *, real: float,
                 window: tuple[int, int], step: int, forbidden: list[int] | None = None,
                 exclude: np.ndarray | None = None, dates: pd.DatetimeIndex | None = None,
                 alpha: float = 0.05, max_n: int = 400) -> PlaceboTime:
    """fit(i)는 위치 i를 시행일로 본 추정. window=(사전 길이, 사후 길이).

    forbidden: 가짜 창 [i − 사전, i + 사후)에 들어 있으면 안 되는 위치(실제 시행일, 계절 메아리).
    exclude: 가짜 시행일 후보에서 뺄 위치 표시(다른 정책 기간 등).
    """
    pre, post = window
    forbidden = [i0] if forbidden is None else forbidden
    cand = list(range(pre // 2, n_index - 1, step))
    if len(cand) > max_n:
        cand = cand[:: int(np.ceil(len(cand) / max_n))]
    eff, ds = [], []
    forb_arr = np.asarray(forbidden, dtype=int)
    for i in cand:
        lo, hi = i - pre, i + post
        if forb_arr.size and np.any((forb_arr >= lo) & (forb_arr < hi)):
            continue
        if exclude is not None and exclude[max(0, lo):min(n_index, hi)].any():
            continue
        e = fit(i)
        if e is None or not np.isfinite(e.effect):
            continue
        eff.append(float(e.effect))
        ds.append(str(dates[i].date()) if dates is not None else str(i))
    a = np.asarray(eff)
    n = len(a)
    if n == 0:
        return PlaceboTime(real, [], [], float("nan"), float("nan"), float("nan"), None, 0, step,
                           "가짜 시행일을 둘 사전 기간이 부족함")
    p_two = (1 + int(np.sum(np.abs(a) >= abs(real)))) / (n + 1)
    p_dir = (1 + int(np.sum(a >= real if real >= 0 else a <= real))) / (n + 1)
    ci = None
    if n >= 19:
        lo_q, hi_q = np.quantile(a, [alpha / 2, 1 - alpha / 2])
        ci = (float(real - hi_q), float(real - lo_q))
    return PlaceboTime(real, eff, ds, p_two, p_dir, float(np.std(a, ddof=1)) if n > 1 else float("nan"),
                       ci, n, step)


@dataclass
class PlaceboSpace:
    real: float
    fakes: dict[str, dict]     # 집단 → {effect, ci, p}
    passed: bool | None
    rule: str

    def to_dict(self) -> dict:
        return asdict(self)


def placebo_space(real: Estimate, fakes: dict[str, Estimate | None], share: float = 0.5) -> PlaceboSpace:
    """가짜 처치 효과가 모두, 실제 효과와 같은 방향으로 실제 효과의 절반(share)보다 작아야 통과.

    대조 집단에만 사건이 생기면 정책 대상이 아닌 집단에서도 비슷한 크기의 '효과'가 나온다. 그 경우를 걸러 낸다.
    (규칙 v1은 '실제 효과가 모든 가짜 효과보다 큼'이었다. 합성 정확도 측정에서 대조 집단 충격을 놓쳐 v2에서 강화)
    """
    ok = {k: v for k, v in fakes.items() if v is not None}
    rule = f"가짜 처치 효과가 모두 실제 효과의 {share:.0%} 미만(같은 방향 기준)"
    if not ok:
        return PlaceboSpace(real.effect, {}, None, rule)
    sgn = 1.0 if real.effect >= 0 else -1.0
    passed = all(sgn * v.effect < share * abs(real.effect) for v in ok.values())
    return PlaceboSpace(real.effect, {k: {"effect": v.effect, "pct": v.pct, "ci": v.ci, "p": v.p,
                                          "significant": v.significant} for k, v in ok.items()},
                        passed, rule)


@dataclass
class Sensitivity:
    specs: dict[str, dict]
    n_pass: int
    n: int
    same_sign: int
    rule: str = "주분석과 같은 방향이고 95% 신뢰구간이 0을 포함하지 않음"

    @property
    def passed(self) -> bool:
        return self.n > 0 and self.n_pass >= int(np.ceil(2 * self.n / 3))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def sensitivity(main: Estimate, specs: dict[str, Estimate | None]) -> Sensitivity:
    sgn = np.sign(main.effect)
    out, n_pass, same = {}, 0, 0
    for k, e in specs.items():
        if e is None:
            out[k] = {"available": False}
            continue
        ok = e.significant and np.sign(e.effect) == sgn
        n_pass += ok
        same += int(np.sign(e.effect) == sgn)
        out[k] = {"available": True, "effect": e.effect, "pct": e.pct, "ci": e.ci, "p": e.p, "pass": bool(ok)}
    n = sum(1 for v in out.values() if v.get("available"))
    return Sensitivity(out, int(n_pass), n, same)


@dataclass
class DoubleCompute:
    quantity: str
    pandas: float
    sql: float
    tol: float
    match: bool

    def to_dict(self) -> dict:
        return asdict(self)


def two_by_two_pandas(df: pd.DataFrame, y: str, unit: str, treated: str, post: str) -> float:
    """단위별 (사후 평균 − 사전 평균)의 처치 평균 − 대조 평균."""
    u = df.groupby([unit, treated, post])[y].mean().unstack(post)
    ch = (u[1] - u[0]).dropna().reset_index()
    ch.columns = [unit, treated, "ch"]
    return float(ch.loc[ch[treated] == 1, "ch"].mean() - ch.loc[ch[treated] == 0, "ch"].mean())


def two_by_two_sql(df: pd.DataFrame, y: str, unit: str, treated: str, post: str) -> float:
    con = duckdb.connect()
    con.register("p", df[[unit, treated, post, y]].rename(columns={unit: "u", treated: "tr", post: "po", y: "y"}))
    q = """
    WITH u AS (
      SELECT u, max(tr) AS tr, avg(y) FILTER (WHERE po = 1) - avg(y) FILTER (WHERE po = 0) AS ch
      FROM p GROUP BY u HAVING count(*) FILTER (WHERE po = 1) > 0 AND count(*) FILTER (WHERE po = 0) > 0
    )
    SELECT avg(ch) FILTER (WHERE tr = 1) - avg(ch) FILTER (WHERE tr = 0) FROM u
    """
    v = con.execute(q).fetchall()[0][0]
    con.close()
    return float(v)


def double_compute_2x2(df: pd.DataFrame, y: str, unit: str, treated: str, post: str,
                       tol: float = 1e-9) -> DoubleCompute:
    a = two_by_two_pandas(df, y, unit, treated, post)
    b = two_by_two_sql(df, y, unit, treated, post)
    return DoubleCompute("2×2 평균 차분(단위 평균)", a, b, tol, bool(abs(a - b) <= tol))


def series_means_sql(g_adj: np.ndarray, i0: int, a: int, b: int, skip: int = 0) -> float:
    """시계열 효과(추세 없음)는 '사후 평균 − 사전 평균'과 같다. SQL로 따로 계산한다."""
    con = duckdb.connect()
    t = pd.DataFrame({"i": np.arange(a, b), "y": g_adj[a:b]})
    con.register("s", t)
    v = con.execute(
        "SELECT avg(y) FILTER (WHERE i >= ? AND i >= ? + ?) - avg(y) FILTER (WHERE i < ?) "
        "FROM s WHERE isfinite(y)", [i0, i0, skip, i0]).fetchall()[0][0]
    con.close()
    return float(v)
