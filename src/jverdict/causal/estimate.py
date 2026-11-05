"""추정 툴: 정책 효과의 크기와 불확실성을 계산한다.

두 경로
1. 집계 시계열 (est.its): 대상 집단과 대조 집단의 합계가 시점별로 하나씩 있을 때 (버스 이용자 유형별 일 합계 등)
       g[t]   = log(1 + 대상[t]) − log(1 + 대조[t])            대조가 없으면 log(1 + 대상[t])
       계절 차분: g_adj[t] = g[t] − median_k g[t − k·주기]       같은 요일·같은 계절의 과거 값과 비교
       회귀:     g_adj[t] = a + β·사후[t] (+ γ·τ + δ·사후·τ)       τ = 정책 시점부터의 연 단위 시간
       효과 θ = β (추세 보정 시 β + δ·평균 τ_사후, 즉 사전 추세 연장선 대비 사후 평균 이탈)
   계절 차분 덕분에 방학·요일처럼 해마다 반복되는 변화와 '꾸준한' 선형 추세는 효과로 잡히지 않는다.
   추론: 일 단위 HAC + fixed-b 임계값, 월 단위 AR(1) 오차 모형. 결측일은 빠지고, 시차는 실제 날짜 간격으로 센다.

2. 단위 패널 (est.did): 집단 × 단위(업종 등) × 시점 패널이 있을 때
       y[i,t] = α_i + λ_t(또는 업종×월 등) + β·처치[i]·사후[t] + ε
   고정효과는 흡수하고, 표준오차는 단위 군집 강건(CR1). 군집이 적으면 와일드 군집 부트스트랩을 함께 계산한다.
   이벤트스터디는 기준 시점을 뺀 모든 시점에 처치×시점 계수를 두고, 사전 계수의 결합 검정을 사전추세 검정으로 쓴다.

모든 효과는 로그 단위다. 백분율 = exp(θ) − 1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from jverdict.quality._panel import FreqParams, fmt, freq_params
from jverdict.quality.breaks import _seasonal_adjust, _series

from scipy import stats as sps

from .stats import FEResult, ar1_lincom, ar1_phi, fe_ols, fixedb_cv, ols_hac, t_ci, wald, wild_cluster

# 기본 창 (시계열): 사전 2년, 사후 6개월. 일 단위 HAC 시차 14일 (합성데이터 보정 결과, docs 참고).
DEFAULTS = {"D": {"pre": 728, "post": 182, "lags": 14, "per_year": 365.25},
            "MS": {"pre": 24, "post": 6, "lags": 3, "per_year": 12.0}}


@dataclass
class Estimate:
    method: str
    effect: float                      # 로그 효과
    se: float
    ci: tuple[float, float]
    p: float
    dof: float
    alpha: float
    n_obs: int
    inference: str
    label: str = ""
    n_pre: int = 0
    n_post: int = 0
    window: tuple[str, str] | None = None
    details: dict = field(default_factory=dict)

    @property
    def pct(self) -> float:
        return float(np.expm1(self.effect))

    @property
    def ci_pct(self) -> tuple[float, float]:
        return float(np.expm1(self.ci[0])), float(np.expm1(self.ci[1]))

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.ci[0]) and (self.ci[0] > 0 or self.ci[1] < 0))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pct"], d["ci_pct"], d["significant"] = self.pct, self.ci_pct, self.significant
        return d

    def short_ko(self) -> str:
        lo, hi = self.ci_pct
        return f"{self.pct * 100:+.1f}% (95% 구간 {lo * 100:+.1f}% ~ {hi * 100:+.1f}%, p={self.p:.3f})"


# =========================================================================== 1. 집계 시계열
@dataclass
class GapSeries:
    """계절 차분까지 끝낸 격차 시계열. 주분석·위약·민감도가 같은 계열을 공유한다."""
    index: pd.DatetimeIndex
    g: np.ndarray
    g_adj: np.ndarray
    fp: FreqParams
    freq: str
    season: str
    season_years: int
    target: list[str]
    reference: list[str]

    def pos(self, date) -> int:
        i = int(self.index.searchsorted(pd.Timestamp(date)))
        if i >= len(self.index):
            raise ValueError(f"정책 시점 {date}이 데이터 기간 뒤에 있음")
        return i


def gap_series(wide: pd.DataFrame, target: str | list[str], reference: str | list[str] | None,
               freq: str, *, season: str = "sdiff", season_years: int = 3,
               mask: np.ndarray | None = None) -> GapSeries:
    fp = freq_params(freq)
    tcols = [target] if isinstance(target, str) else list(target)
    rcols = [] if reference is None else ([reference] if isinstance(reference, str) else list(reference))
    g = np.log1p(_series(wide, tcols))
    if rcols:
        g = g - np.log1p(_series(wide, rcols))
    g = g.astype(float)
    if mask is not None:
        g[np.asarray(mask, bool)] = np.nan
    if season == "sdiff":
        adj = _seasonal_adjust(g, replace(fp, years_back=season_years))
    elif season == "none":
        adj = g.copy()
    else:
        raise ValueError(f"알 수 없는 계절 처리: {season}")
    return GapSeries(wide.index, g, adj, fp, freq, season, season_years, tcols, rcols)


def _infer(gs: GapSeries, i0: int, y: np.ndarray, X: np.ndarray, pos: np.ndarray, c: np.ndarray,
           n_seg: int, alpha: float, lags: int | None) -> tuple[float, float, tuple[float, float], float, float, str]:
    """시계열 효과의 추론.

    일 단위: HAC(Bartlett) + fixed-b 임계값. b = 시차 / 효과 구간 관측 수.
        사후 더미가 구간 안 잔차 합을 0으로 묶어 HAC가 분산을 작게 잡는 편향을 보정한다.
    월 단위: AR(1) 오차 모형. 계수는 정책 전 전체 기간의 계절 차분 격차로 추정(Kendall 보정).
        관측이 수십 개뿐이라 HAC가 크게 무너지는 경우(합성데이터 실측 포함률 50~65%)를 막는다.
    """
    if gs.freq == "D":
        L = DEFAULTS["D"]["lags"] if lags is None else lags
        r = ols_hac(y, X, pos, L)
        eff, se = r.lincom(c)
        cv = fixedb_cv(L / max(n_seg, 1), alpha)
        z = abs(eff) / se if se > 0 else np.inf
        p = float(2 * sps.norm.sf(z * sps.norm.ppf(1 - alpha / 2) / cv))   # 임계값 축척으로 근사한 p
        return eff, se, (eff - cv * se, eff + cv * se), p, r.dof, f"HAC(시차 {r.lags}일) + fixed-b 임계값 {cv:.2f}"
    phi, n_phi = ar1_phi(gs.g_adj[:i0])
    eff, se, dof = ar1_lincom(y, X, pos, c, phi)
    ci, p = t_ci(eff, se, dof, alpha)
    return eff, se, ci, p, dof, f"AR(1) 오차(φ={phi:.2f}, 정책 전 {n_phi}개월로 추정), t({dof})"


def its_fit(gs: GapSeries, i0: int, *, pre: int | None = None, post: int | None = None, skip: int = 0,
            trend: bool = False, lags: int | None = None, alpha: float = 0.05,
            min_frac: float = 0.5, label: str = "") -> Estimate | None:
    """정책 위치 i0 기준 단절 시계열. 관측이 창의 min_frac 미만이면 None."""
    dflt = DEFAULTS["D" if gs.freq == "D" else "MS"]
    pre = dflt["pre"] if pre is None else pre
    post = dflt["post"] if post is None else post
    if gs.season == "sdiff":
        post = min(post, gs.fp.season)                 # 사후가 한 주기를 넘으면 계절 기준이 오염된다
    a, b = max(0, i0 - pre), min(len(gs.index), i0 + post)
    pos = np.arange(a, b)
    y = gs.g_adj[a:b]
    is_post = pos >= i0
    keep = np.isfinite(y) & ~(is_post & (pos < i0 + skip))
    n_pre, n_post = int(np.sum(keep & ~is_post)), int(np.sum(keep & is_post))
    span_pre, span_post = i0 - a, b - i0 - skip
    if span_pre < min_frac * pre or span_post < 1:
        return None
    if n_pre < max(3, min_frac * span_pre) or n_post < max(1, min_frac * span_post):
        return None
    if trend and n_pre < 6:
        return None
    pos, y, is_post = pos[keep], y[keep], is_post[keep].astype(float)
    tau = (pos - i0) / dflt["per_year"]
    cols = [np.ones_like(y), is_post]
    if trend:
        cols += [tau, is_post * tau]
    X = np.column_stack(cols)
    c = np.zeros(X.shape[1])
    c[1] = 1.0
    if trend:
        c[3] = float(tau[is_post == 1].mean())
    try:
        eff, se, ci, p, dof, inf = _infer(gs, i0, y, X, pos, c, n_post, alpha, lags)
    except (ValueError, np.linalg.LinAlgError):
        return None
    idx = gs.index
    det = {"pre_mean": float(np.mean(y[is_post == 0])), "post_mean": float(np.mean(y[is_post == 1])),
           "season": gs.season, "season_years": gs.season_years, "trend": trend, "skip": skip,
           "target": gs.target, "reference": gs.reference, "policy": fmt(idx[i0], gs.freq)}
    if trend:
        bb = np.linalg.lstsq(X, y, rcond=None)[0]
        det["pre_slope_per_year"] = float(bb[2])
    return Estimate("its", eff, se, ci, p, dof, alpha, len(y), inf, label, n_pre, n_post,
                    (fmt(idx[pos[0]], gs.freq), fmt(idx[pos[-1]], gs.freq)), det)


def its(wide: pd.DataFrame, target, reference, date, freq: str, *, season: str = "sdiff",
        season_years: int = 3, mask=None, **kw) -> Estimate | None:
    gs = gap_series(wide, target, reference, freq, season=season, season_years=season_years, mask=mask)
    return its_fit(gs, gs.pos(date), **kw)


def series_pretrend(gs: GapSeries, i0: int, *, pre: int | None = None, lags: int | None = None) -> dict:
    """사전 기간 격차(계절 차분)의 선형 추세 검정. 평행추세 가정의 점검. 추론은 its_fit과 같다."""
    dflt = DEFAULTS["D" if gs.freq == "D" else "MS"]
    pre = dflt["pre"] if pre is None else pre
    a = max(0, i0 - pre)
    pos = np.arange(a, i0)
    y = gs.g_adj[a:i0]
    ok = np.isfinite(y)
    if ok.sum() < 6:
        return {"available": False, "reason": "사전 관측 부족", "p": None}
    pos, y = pos[ok], y[ok]
    tau = (pos - i0) / dflt["per_year"]
    X = np.column_stack([np.ones_like(y), tau])
    s, se, _, p, _, inf = _infer(gs, i0, y, X, pos, np.array([0.0, 1.0]), len(y), 0.05, lags)
    span_years = float(tau.max() - tau.min())
    return {"available": True, "method": f"사전 기간 선형 추세({inf})", "slope_per_year": s, "se": se, "p": p,
            "n": int(len(y)), "drift_over_pre": s * span_years}


# =========================================================================== 2. 단위 패널
def _fe_arrays(df: pd.DataFrame, fe: list[str]) -> list[np.ndarray]:
    return [df[c].to_numpy() for c in fe]


def did(df: pd.DataFrame, y: str, unit: str, time: str, treated: str, date, *, fe: list[str] | None = None,
        cluster: str | None = None, weights: str | None = None, alpha: float = 0.05,
        bootstrap: bool | None = None, label: str = "") -> Estimate:
    """이중차분(고정효과 흡수). treated는 0/1 열. 사후 = time ≥ date."""
    fe = fe or [unit, time]
    cluster = cluster or unit
    d = df.dropna(subset=[y]).reset_index(drop=True)
    post = (pd.to_datetime(d[time]) >= pd.Timestamp(date)).to_numpy()
    tr = d[treated].to_numpy().astype(bool)
    x = (tr & post).astype(float)
    w = d[weights].to_numpy(float) if weights else None
    res = fe_ols(d[y].to_numpy(float), x[:, None], ["tp"], _fe_arrays(d, fe), d[cluster].to_numpy(), w)
    b, se = res.coef("tp")
    ci, p = t_ci(b, se, res.dof, alpha)
    det = {"fe": fe, "cluster": cluster, "weights": weights, "n_units": int(d[unit].nunique()),
           "n_treat_units": int(d.loc[tr, unit].nunique()), "n_clusters": res.n_clusters,
           "policy": str(pd.Timestamp(date).date())}
    if bootstrap is None:
        bootstrap = res.n_clusters < 50
    if bootstrap:
        wb = wild_cluster(res, "tp", alpha=alpha)
        det["wild_bootstrap"] = {"p": wb["p"], "ci": wb["ci"], "B": wb["B"], "weights": wb["weights"],
                                 "treated_clusters": wb["treated_clusters"]}
    times = pd.to_datetime(d[time])
    return Estimate("did", b, se, ci, p, res.dof, alpha, res.n, f"CR1(군집 {res.n_clusters}개, t 자유도 {res.dof})",
                    label, int((~post).sum()), int(post.sum()),
                    (str(times.min().date()), str(times.max().date())), det)


@dataclass
class EventStudy:
    table: pd.DataFrame        # time, rel, coef, se, ci_lo, ci_hi, pre
    ref: str
    pretrend_F: float | None
    pretrend_p: float | None
    pretrend_q: int
    n_clusters: int

    def to_dict(self) -> dict:
        return {"ref": self.ref, "pretrend_F": self.pretrend_F, "pretrend_p": self.pretrend_p,
                "pretrend_q": self.pretrend_q, "n_clusters": self.n_clusters,
                "table": self.table.assign(time=self.table.time.astype(str)).to_dict("records")}


def event_study(df: pd.DataFrame, y: str, unit: str, time: str, treated: str, date, ref, *,
                fe: list[str] | None = None, cluster: str | None = None, weights: str | None = None,
                alpha: float = 0.05) -> EventStudy:
    fe = fe or [unit, time]
    cluster = cluster or unit
    d = df.dropna(subset=[y]).reset_index(drop=True)
    times = pd.to_datetime(d[time])
    tr = d[treated].to_numpy().astype(bool)
    ref = pd.Timestamp(ref)
    periods = sorted(t for t in times.unique() if t != ref)
    names = [f"e{k}" for k in range(len(periods))]
    X = np.column_stack([(tr & (times == t).to_numpy()).astype(float) for t in periods])
    w = d[weights].to_numpy(float) if weights else None
    res = fe_ols(d[y].to_numpy(float), X, names, _fe_arrays(d, fe), d[cluster].to_numpy(), w)
    rows = []
    for nm, t in zip(names, periods):
        b, se = res.coef(nm)
        ci, _ = t_ci(b, se, res.dof, alpha)
        rows.append({"time": t, "coef": b, "se": se, "ci_lo": ci[0], "ci_hi": ci[1],
                     "pre": bool(t < pd.Timestamp(date))})
    tab = pd.DataFrame(rows)
    leads = [nm for nm, t in zip(names, periods) if t < pd.Timestamp(date)]
    F, p, q = wald(res, leads) if leads else (None, None, 0)
    return EventStudy(tab, str(ref.date()), F, p, q, res.n_clusters)


def did_trend(df: pd.DataFrame, y: str, unit: str, time: str, treated: str, date, *,
              fe: list[str] | None = None, cluster: str | None = None, weights: str | None = None,
              alpha: float = 0.05, label: str = "추세 보정") -> Estimate | None:
    """처치 집단 고유의 선형 추세를 사전 기간에서만 추정해 연장한 뒤, 사후 평균 이탈을 효과로 본다.

    y = FE + γ·처치·τ + 처치·사후·(β + δ·τ),  θ = β + δ·평균 τ(처치·사후 관측)
    사후 기울기를 따로 두므로 γ는 사전 기간 자료로만 정해진다.
    """
    fe = fe or [unit, time]
    cluster = cluster or unit
    d = df.dropna(subset=[y]).reset_index(drop=True)
    t0 = pd.Timestamp(date)
    times = pd.to_datetime(d[time])
    tau = ((times - t0).dt.days / 365.25).to_numpy()
    tr = d[treated].to_numpy().astype(float)
    post = (times >= t0).to_numpy().astype(float)
    if len(np.unique(times[(tr == 1) & (post == 0)])) < 2:
        return None
    X = np.column_stack([tr * tau, tr * post, tr * post * tau])
    w = d[weights].to_numpy(float) if weights else None
    res = fe_ols(d[y].to_numpy(float), X, ["trend", "tp", "tp_tau"], _fe_arrays(d, fe), d[cluster].to_numpy(), w)
    mt = float(tau[(tr == 1) & (post == 1)].mean())
    eff, se = res.lincom({"tp": 1.0, "tp_tau": mt})
    ci, p = t_ci(eff, se, res.dof, alpha)
    g, gse = res.coef("trend")
    return Estimate("did_trend", eff, se, ci, p, res.dof, alpha, res.n,
                    f"CR1(군집 {res.n_clusters}개)", label, int((post == 0).sum()), int((post == 1).sum()),
                    (str(times.min().date()), str(times.max().date())),
                    {"pre_slope_per_year": g, "pre_slope_se": gse, "mean_tau_post": mt})


def fit_for_tests(df, y, unit, time, treated, date, *, fe=None, cluster=None, weights=None) -> FEResult:
    """테스트·이중 계산용: 원시 결과 객체를 돌려준다."""
    fe = fe or [unit, time]
    d = df.dropna(subset=[y]).reset_index(drop=True)
    post = (pd.to_datetime(d[time]) >= pd.Timestamp(date)).to_numpy()
    x = (d[treated].to_numpy().astype(bool) & post).astype(float)
    w = d[weights].to_numpy(float) if weights else None
    return fe_ols(d[y].to_numpy(float), x[:, None], ["tp"], _fe_arrays(d, fe), d[cluster or unit].to_numpy(), w)


# =========================================================================== 3. 집단 수준 추론 (처치 집단이 하나일 때)
def panel_gap(d: pd.DataFrame, y: str, unit: str, time: str, trcol: str, t0) -> pd.Series:
    """처치 단위 평균 − 대조 단위 평균 (각 단위의 사전 평균을 뺀 값)의 시점별 격차.

    단위·시점 고정효과 이중차분은 균형 패널에서 '사후 평균 격차 − 사전 평균 격차'와 같다.
    """
    t0 = pd.Timestamp(t0)
    pre = d[pd.to_datetime(d[time]) < t0]
    um = pre.groupby(unit)[y].mean()
    dd = d[d[unit].isin(um.index)]
    r = dd[y] - dd[unit].map(um)
    g = r.groupby([pd.to_datetime(dd[time]), dd[trcol]]).mean().unstack()
    if 1 not in g or 0 not in g:
        return pd.Series(dtype=float)
    return (g[1] - g[0]).dropna()


def group_level(d_window: pd.DataFrame, d_full: pd.DataFrame, y: str, unit: str, time: str, trcol: str,
                t0, alpha: float = 0.05) -> dict | None:
    """처치가 집단 단위로 배정되고 처치 집단이 하나일 때의 추론 (Donald·Lang식 집계).

    같은 집단의 업종들은 집단 공통 충격을 함께 받으므로, 업종 군집 표준오차는 지나치게 좁다
    (합성 패널 실측 포함률 20%). 집단 수준 격차 시계열에 AR(1) 오차를 두고 표준오차를 다시 계산한다.
    AR 계수는 정책 전 전체 기간의 격차로 추정한다(Kendall 보정).
    """
    t0 = pd.Timestamp(t0)
    gw = panel_gap(d_window, y, unit, time, trcol, t0)
    gf = panel_gap(d_full, y, unit, time, trcol, t0)
    if len(gw) < 4 or (gw.index >= t0).sum() < 1 or (gw.index < t0).sum() < 2:
        return None
    tf = pd.DatetimeIndex(sorted(pd.to_datetime(d_full[time]).unique()))
    arr = np.full(len(tf), np.nan)
    gpre = gf[gf.index < t0]
    arr[tf.get_indexer(gpre.index)] = gpre.to_numpy()
    phi, nphi = ar1_phi(arr)
    pos = tf.get_indexer(gw.index)
    yv = gw.to_numpy(float)
    post = (gw.index >= t0).astype(float)
    X = np.column_stack([np.ones_like(yv), post])
    eff, se, dof = ar1_lincom(yv, X, pos, np.array([0.0, 1.0]), phi)
    out = {"effect": eff, "se": se, "dof": dof, "phi": phi, "n_phi": nphi, "n_periods": int(len(gw))}
    pre_m = post == 0
    if pre_m.sum() >= 6:
        i0 = int(tf.searchsorted(t0))
        tau = (pos[pre_m] - i0) / 12.0
        Xp = np.column_stack([np.ones(pre_m.sum()), tau])
        s, sse, sdof = ar1_lincom(yv[pre_m], Xp, pos[pre_m], np.array([0.0, 1.0]), phi)
        _, sp = t_ci(s, sse, sdof)
        out["pretrend"] = {"slope_per_year": s, "se": sse, "p": sp, "n": int(pre_m.sum())}
        tau_all = (pos - i0) / 12.0
        Xt = np.column_stack([np.ones_like(yv), post, tau_all, post * tau_all])
        c = np.array([0.0, 1.0, 0.0, float(tau_all[post == 1].mean())])
        te, tse, tdof = ar1_lincom(yv, Xt, pos, c, phi)
        out["trend_adjusted"] = {"effect": te, "se": tse, "dof": tdof}
    return out


def apply_group_level(est: Estimate, gl: dict | None) -> Estimate:
    """군집 표준오차와 집단 수준 표준오차 중 큰 쪽으로 구간을 다시 만든다."""
    if gl is None or not np.isfinite(gl["se"]):
        return est
    est.details["cr1_se"], est.details["group_level"] = est.se, {k: v for k, v in gl.items()
                                                                 if k not in ("pretrend", "trend_adjusted")}
    if gl["se"] > est.se:
        est.se = gl["se"]
        est.ci, est.p = t_ci(est.effect, est.se, gl["dof"], est.alpha)
        est.dof = gl["dof"]
        phi_txt = f"φ={gl['phi']:.2f}" if gl["n_phi"] >= 8 else f"정책 전 {gl['n_phi']}시점이라 φ 추정 불가 → 0으로 둠"
        est.inference = (f"집단 수준 격차 시계열 표준오차({phi_txt}, 격차 {gl['n_periods']}시점) — "
                         f"업종 군집 표준오차 {est.details['cr1_se']:.3f}보다 커서 채택")
    return est
