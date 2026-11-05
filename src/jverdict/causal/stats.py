"""인과 엔진의 통계 계산 핵심부 (numpy만 사용, 외부 추정 패키지에 의존하지 않음).

- ols_hac        : 시계열 회귀 + Newey-West(HAC) 공분산. 결측으로 빠진 시점이 있어도 실제 시차로 계산한다.
- absorb         : 고정효과 흡수(교대 사영). 불균형 패널도 처리한다.
- fe_ols         : 고정효과 흡수 회귀 + 군집 강건 공분산(CR1). 계수 여러 개, 가중치 지원.
- wald           : 선형 제약 결합 검정 (F 분포, 자유도 = 군집 수 − 1).
- wild_cluster   : 제약 와일드 군집 부트스트랩(WCR) p값과 검정 역산 신뢰구간. 군집이 적을 때 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------- 시계열 회귀
@dataclass
class OLSResult:
    beta: np.ndarray
    cov: np.ndarray
    resid: np.ndarray
    n: int
    k: int
    dof: int
    lags: int

    def lincom(self, c: np.ndarray) -> tuple[float, float]:
        c = np.asarray(c, dtype=float)
        return float(c @ self.beta), float(np.sqrt(max(c @ self.cov @ c, 0.0)))


def ols_hac(y: np.ndarray, X: np.ndarray, pos: np.ndarray, lags: int) -> OLSResult:
    """y = Xb + e, Newey-West(Bartlett) 공분산.

    pos는 각 행의 시간 위치(정수). 결측으로 빠진 시점이 있어도 시차는 실제 시간 간격으로 센다.
    작은 표본 보정 n/(n−k)를 곱한다.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    if n <= k:
        raise ValueError(f"관측 {n}개로 계수 {k}개를 추정할 수 없음")
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    pos = np.asarray(pos, int)
    span = int(pos.max() - pos.min() + 1)
    H = np.zeros((span, k))
    H[pos - pos.min()] = X * u[:, None]
    S = H.T @ H
    L = int(max(0, min(lags, span - 1)))
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        G = H[lag:].T @ H[:-lag]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv * (n / (n - k))
    return OLSResult(b, cov, u, n, k, n - k, L)


# --------------------------------------------------------------------------- 고정효과 흡수
def _codes(a) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray(a), return_inverse=True)
    return inv.astype(np.int64), int(inv.max()) + 1 if len(inv) else 0


def absorb(M: np.ndarray, fes: list[np.ndarray], w: np.ndarray | None = None,
           tol: float = 1e-11, maxit: int = 5000) -> np.ndarray:
    """M의 각 열에서 고정효과들을 (가중) 교대 사영으로 제거한다."""
    M = np.array(M, dtype=float, copy=True)
    if M.ndim == 1:
        M = M[:, None]
    if not fes:
        return M
    w = np.ones(len(M)) if w is None else np.asarray(w, float)
    coded = [_codes(f) for f in fes]
    wsum = [np.bincount(c, weights=w, minlength=g) for c, g in coded]
    scale = np.maximum(np.abs(M).max(axis=0), 1.0)
    for _ in range(maxit):
        delta = 0.0
        for (c, g), ws in zip(coded, wsum):
            for j in range(M.shape[1]):
                m = np.bincount(c, weights=w * M[:, j], minlength=g) / np.where(ws > 0, ws, 1)
                M[:, j] -= m[c]
                delta = max(delta, float(np.abs(m).max() / scale[j]))
        if len(coded) == 1 or delta < tol:
            break
    return M


def fe_dof(fes: list[np.ndarray]) -> int:
    """흡수한 고정효과의 자유도 (연결된 패널 가정: 수준 수 합 − (고정효과 수 − 1))."""
    if not fes:
        return 1                                   # 절편
    return sum(_codes(f)[1] for f in fes) - (len(fes) - 1)


@dataclass
class FEResult:
    names: list[str]
    beta: np.ndarray
    cov: np.ndarray
    n: int
    k_total: int
    n_clusters: int
    Xt: np.ndarray          # 고정효과를 흡수한 설명변수
    yt: np.ndarray          # 고정효과를 흡수한 결과변수
    w: np.ndarray
    clusters: np.ndarray
    X: np.ndarray           # 원래 설명변수 (처치 군집 수 계산용)

    def idx(self, name: str) -> int:
        return self.names.index(name)

    def coef(self, name: str) -> tuple[float, float]:
        i = self.idx(name)
        return float(self.beta[i]), float(np.sqrt(self.cov[i, i]))

    def lincom(self, c: dict[str, float]) -> tuple[float, float]:
        v = np.zeros(len(self.names))
        for k, x in c.items():
            v[self.idx(k)] = x
        return float(v @ self.beta), float(np.sqrt(max(v @ self.cov @ v, 0.0)))

    @property
    def dof(self) -> int:
        return max(self.n_clusters - 1, 1)


def _cr1(Xt, u, w, clusters, XtWX_inv, n, k_total) -> np.ndarray:
    gcode, G = _codes(clusters)
    scores = Xt * (w * u)[:, None]
    Sg = np.zeros((G, Xt.shape[1]))
    np.add.at(Sg, gcode, scores)
    meat = Sg.T @ Sg
    c = (G / max(G - 1, 1)) * ((n - 1) / max(n - k_total, 1))
    return c * XtWX_inv @ meat @ XtWX_inv


def fe_ols(y: np.ndarray, X: np.ndarray, names: list[str], fes: list[np.ndarray],
           clusters: np.ndarray, w: np.ndarray | None = None) -> FEResult:
    """고정효과 흡수 OLS(가중 가능) + CR1 군집 강건 공분산.

    자유도 보정 G/(G−1)·(N−1)/(N−K)의 K는 흡수한 고정효과 자유도를 포함한다(더미 회귀와 같은 값).
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, float)
    Z = absorb(np.column_stack([y, X]), fes, w)
    yt, Xt = Z[:, 0], Z[:, 1:]
    XtWX = Xt.T @ (Xt * w[:, None])
    XtWX_inv = np.linalg.pinv(XtWX)
    b = XtWX_inv @ (Xt.T @ (w * yt))
    u = yt - Xt @ b
    k_total = Xt.shape[1] + fe_dof(fes)
    cov = _cr1(Xt, u, w, clusters, XtWX_inv, n, k_total)
    return FEResult(list(names), b, cov, n, k_total, _codes(clusters)[1], Xt, yt, w, np.asarray(clusters), X)


def wald(res: FEResult, names: list[str]) -> tuple[float, float, int]:
    """H0: 지정한 계수가 모두 0. F = b'V⁻¹b/q, 자유도 (q, G−1)."""
    ii = [res.idx(n) for n in names]
    b = res.beta[ii]
    V = res.cov[np.ix_(ii, ii)]
    q = len(ii)
    F = float(b @ np.linalg.pinv(V) @ b / q)
    p = float(sps.f.sf(F, q, res.dof))
    return F, p, q


# --------------------------------------------------------------------------- 와일드 군집 부트스트랩
_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def wild_cluster(res: FEResult, name: str, *, B: int = 9999, seed: int = 20261103,
                 alpha: float = 0.05) -> dict:
    """제약 와일드 군집 부트스트랩(WCR, MacKinnon·Webb).

    대상 계수 외의 설명변수까지 흡수한 FWL 형태(x̃, ỹ)에서 계산하므로 반복마다 회귀를 다시 풀 필요가 없다.
    군집 12개 이하면 Webb 6점 가중치, 아니면 Rademacher. 처치 군집이 1~2개면 이 방법도 믿기 어렵다(결과에 표시).
    """
    i = res.idx(name)
    others = [j for j in range(len(res.names)) if j != i]
    w = res.w
    x = res.Xt[:, i]
    yv = res.yt
    if others:
        O = res.Xt[:, others]
        P = np.linalg.pinv(O.T @ (O * w[:, None])) @ (O.T * w)
        x = x - O @ (P @ x)
        yv = yv - O @ (P @ yv)
    g, G = _codes(res.clusters)
    sw = np.sqrt(w)
    x, yv = x * sw, yv * sw                              # 가중 회귀를 비가중 형태로
    A = np.bincount(g, weights=x * x, minlength=G)
    S = A.sum()
    bhat = float((x @ yv) / S)
    c = (G / max(G - 1, 1)) * ((res.n - 1) / max(res.n - res.k_total, 1))
    u_hat = yv - bhat * x
    se_hat = float(np.sqrt(c * np.sum(np.bincount(g, weights=x * u_hat, minlength=G) ** 2) / S ** 2))
    rng = np.random.default_rng(seed)
    V = rng.choice(_WEBB, size=(B, G)) if G <= 12 else rng.choice([-1.0, 1.0], size=(B, G))
    nz = np.bincount(g, weights=(res.X[:, i] != 0).astype(float), minlength=G)
    treated_clusters = int(np.sum(nz > 0))

    def pval(b0: float) -> float:
        u0 = yv - b0 * x
        Bg = np.bincount(g, weights=x * u0, minlength=G)
        d = (V @ Bg) / S                                  # β* − β0
        s = V * Bg[None, :] - d[:, None] * A[None, :]     # 군집 점수
        se = np.sqrt(c * np.sum(s * s, axis=1)) / S
        t_star = d / np.where(se > 0, se, np.inf)
        t_obs = (bhat - b0) / se_hat if se_hat > 0 else 0.0
        return float((1 + np.sum(np.abs(t_star) >= abs(t_obs))) / (B + 1))

    def bound(sign: int) -> float:
        lo, hi = bhat, bhat + sign * 4 * se_hat
        k = 0
        while pval(hi) > alpha and k < 8:                 # 기각되는 지점을 찾을 때까지 넓힌다
            hi = bhat + sign * (4 + 4 * (k + 1)) * se_hat
            k += 1
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if pval(mid) > alpha:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    return {"p": pval(0.0), "ci": (bound(-1), bound(+1)), "B": B, "clusters": G,
            "weights": "Webb 6점" if G <= 12 else "Rademacher", "beta": bhat, "se_cr1": se_hat,
            "treated_clusters": treated_clusters}


def t_ci(est: float, se: float, dof: float, alpha: float = 0.05) -> tuple[tuple[float, float], float]:
    """t 분포 신뢰구간과 양측 p값."""
    if not np.isfinite(se) or se <= 0:
        return (float("nan"), float("nan")), float("nan")
    q = float(sps.t.ppf(1 - alpha / 2, dof))
    p = float(2 * sps.t.sf(abs(est / se), dof))
    return (est - q * se, est + q * se), p


# --------------------------------------------------------------------------- 시계열 추론 보정
def fixedb_cv(b: float, alpha: float = 0.05) -> float:
    """Bartlett 커널 HAC의 fixed-b 임계값 (Kiefer·Vogelsang 2005의 다항 근사). b = 시차 / 표본 길이."""
    b = float(np.clip(b, 0.0, 1.0))
    if abs(alpha - 0.10) < 1e-9:
        return 1.6449 + 2.1859 * b + 0.3142 * b ** 2 - 0.3427 * b ** 3
    if abs(alpha - 0.01) < 1e-9:
        return 2.5758 + 4.5154 * b + 0.3416 * b ** 2 - 0.4059 * b ** 3
    return 1.9600 + 2.9694 * b + 0.4160 * b ** 2 - 0.5324 * b ** 3


def ar1_phi(x: np.ndarray) -> tuple[float, int]:
    """평균을 뺀 계열의 1차 자기상관(연속 관측 쌍만), Kendall 편향 보정 (1 + 3φ)/n."""
    pos = np.flatnonzero(np.isfinite(x))
    if len(pos) < 8:
        return 0.0, len(pos)
    u = x[pos] - x[pos].mean()
    d = np.diff(pos) == 1
    den = float(np.sum(u[:-1][d] ** 2))
    phi = float(np.sum(u[1:][d] * u[:-1][d]) / den) if den > 0 else 0.0
    phi += (1 + 3 * phi) / len(u)
    return float(np.clip(phi, 0.0, 0.98)), len(u)


def ar1_lincom(y: np.ndarray, X: np.ndarray, pos: np.ndarray, c: np.ndarray, phi: float) -> tuple[float, float, int]:
    """오차가 AR(1)(계수 phi)일 때 c'b의 표준오차. 잔차 분산은 AR 구조에서 불편이 되도록 보정한다."""
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    u = y - X @ b
    R = phi ** np.abs(pos[:, None] - pos[None, :]).astype(float)
    M = np.eye(n) - X @ XtXi @ X.T
    s2 = float(np.sum(u ** 2) / max(np.trace(M @ R), 1e-12))
    w = (XtXi @ X.T).T @ c
    return float(c @ b), float(np.sqrt(max(s2 * (w @ R @ w), 0.0))), n - k
