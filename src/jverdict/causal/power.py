"""검정력 툴: 지금 데이터로 구분할 수 있는 최소 효과(MDE)와, 기준 효과를 구분하려면 필요한 기간.

MDE = (z_{1−α/2} + z_{검정력}) × 표준오차  (α=0.05, 검정력 80%면 2.80 × SE)
표준오차는 분석적 SE와 위약 분포 표준편차 중 큰 값을 쓴다(자기상관·모형 오차를 과소평가하지 않기 위해).
필요 기간은 SE가 사후 관측 수의 제곱근에 반비례한다고 보고 근사한다(자기상관이 강하면 더 길어질 수 있음).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats as sps


@dataclass
class Power:
    se_used: float
    se_source: str
    mde_log: float
    mde_pct: float
    reference_pct: float          # 기준 효과(정책적으로 의미 있는 최소 크기)
    enough: bool                  # MDE ≤ 기준 효과
    n_post: int
    post_needed: int | None       # 기준 효과를 구분하는 데 필요한 사후 기간 수 (근사)
    unit: str
    # 사전 기간에서 크게 벗어난 시점(로버스트 z > 5). 있으면 표준오차가 큰 이유를 '기간 부족'이 아니라 이 시점으로 설명한다.
    pre_outliers: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def se_ratio(self) -> float:
        """기준 효과를 구분하려면 표준오차를 몇 배 줄여야 하는가."""
        ref_log = float(np.log1p(self.reference_pct))
        return float(self.mde_log / ref_log) if ref_log > 0 else float("nan")

    def outlier_text(self) -> str:
        return ", ".join(f"{o['time']}(대상/대조 로그 격차가 사전 중앙값보다 {abs(o['dev']):.2f} {'낮음' if o['dev'] < 0 else '높음'})"
                         for o in self.pre_outliers)

    def sentence_ko(self) -> str:
        s = f"현재 구분 가능한 최소 효과는 ±{self.mde_pct * 100:.1f}%입니다(기준 효과 {self.reference_pct * 100:.0f}%)."
        if not self.enough and self.pre_outliers:
            s += (f" 표준오차가 큰 주된 이유는 사전 기간에 크게 벗어난 시점이 있기 때문입니다: {self.outlier_text()}."
                  " 기준선이 흔들려 사후 기간을 늘리는 것만으로는 해결하기 어렵습니다"
                  + (f"(약 {self.post_needed}{self.unit} 필요)" if self.post_needed else "")
                  + ". 그 시점에 다른 정책·영업 중단·집계 변경이 있었는지 먼저 확인해야 합니다.")
            return s
        if not self.enough and self.post_needed:
            s += (f" 기준 효과를 구분하려면 표준오차가 지금의 1/{self.se_ratio:.1f} 이하여야 합니다. "
                  f"정책이 계속되는 경우 사후 기간 약 {self.post_needed}{self.unit}(현재 {self.n_post}{self.unit}), "
                  f"끝난 정책이면 더 세밀한 자료(일 단위·전체 카드사 등)가 필요합니다.")
        return s

    def needed_ko(self) -> str:
        if self.pre_outliers:
            return (f"사전 기간 이상 시점({', '.join(o['time'] for o in self.pre_outliers)})에 무슨 일이 있었는지 확인할 자료"
                    "(다른 정책·영업 중단·집계 기준 변경 기록). 원인이 확인되면 그 시점을 뺀 설계를 새로 등록")
        return (f"표준오차를 1/{self.se_ratio:.1f}로 줄일 자료: 사후 약 {self.post_needed}{self.unit}의 관측"
                f"(정책이 계속되는 경우) 또는 더 세밀한 단위의 자료")


def mde(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    return float((sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)) * se)


def pre_outliers(x: np.ndarray, times: list[str], z: float = 5.0, top: int = 3, share: float = 0.5) -> list[dict]:
    """사전 기간 계열에서 중앙값으로부터 로버스트 z(MAD 기준)가 z를 넘는 시점. 큰 순서로 최대 top개.

    그 시점들이 사전 기간 변동(중앙값 기준 제곱합)의 share 이상을 차지할 때만 돌려준다.
    일 단위의 명절 하루처럼 튀지만 표준오차를 좌우하지 않는 시점은 설명 대상이 아니다.
    """
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 6:
        return []
    med = float(np.median(x[ok]))
    mad = 1.4826 * float(np.median(np.abs(x[ok] - med)))
    if mad <= 0:
        return []
    r = [(abs(v - med) / mad, t, v - med) for v, t, k in zip(x, times, ok) if k and abs(v - med) / mad > z]
    total = float(np.sum((x[ok] - med) ** 2))
    if not r or total <= 0 or sum(d * d for _, _, d in r) / total < share:
        return []
    r.sort(reverse=True)
    return [{"time": t, "dev": round(float(d), 4), "robust_z": round(float(zz), 2)} for zz, t, d in r[:top]]


def assess(se_analytic: float, se_placebo: float | None, reference_pct: float, n_post: int, unit: str,
           alpha: float = 0.05, power: float = 0.8, outliers: list | None = None) -> Power:
    cands = [(se_analytic, "분석적 표준오차")]
    if se_placebo is not None and np.isfinite(se_placebo):
        cands.append((se_placebo, "위약 분포 표준편차"))
    se, src = max(cands, key=lambda c: c[0] if np.isfinite(c[0]) else -1)
    m = mde(se, alpha, power)
    ref_log = float(np.log1p(reference_pct))
    enough = bool(m <= ref_log)
    need = None
    if not enough and n_post > 0 and ref_log > 0:
        need = int(np.ceil(n_post * (m / ref_log) ** 2))
    return Power(se, src, m, float(np.expm1(m)), reference_pct, enough, n_post, need, unit, list(outliers or []))
