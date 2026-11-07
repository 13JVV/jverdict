"""검정력 툴: 지금 데이터로 구분할 수 있는 최소 효과(MDE)와, 기준 효과를 구분하려면 필요한 기간.

MDE = (z_{1−α/2} + z_{검정력}) × 표준오차  (α=0.05, 검정력 80%면 2.80 × SE)
표준오차는 분석적 SE와 위약 분포 표준편차 중 큰 값을 쓴다(자기상관·모형 오차를 과소평가하지 않기 위해).
필요 기간: 효과 = 사후 평균 − 사전 평균이므로 분산은 (1/n_사전 + 1/n_사후)에 비례한다(자기상관 배수는 양쪽에 같다고 근사).
사후 기간을 무한히 늘려도 사전 평균의 분산은 남으므로, 사전 기간이 짧으면 기간만으로는 도달할 수 없다(post_reachable=False).
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
    post_reachable: bool | None = None   # False면 사후 기간을 아무리 늘려도 기준 효과를 구분할 수 없음
    n_pre: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def se_ratio(self) -> float:
        """기준 효과를 구분하려면 표준오차를 몇 배 줄여야 하는가."""
        ref_log = float(np.log1p(self.reference_pct))
        return float(self.mde_log / ref_log) if ref_log > 0 else float("nan")

    def outlier_text(self) -> str:
        return ", ".join(f"{o['time']}(대상/대조 격차가 사전 기간 보통 때의 {ratio_text(o['dev'])})" for o in self.pre_outliers)

    def mde_text(self) -> str:
        return mde_text(self.mde_log)

    def sentence_ko(self) -> str:
        s = f"현재 구분 가능한 최소 효과는 {self.mde_text()}입니다(기준 효과 {self.reference_pct * 100:.0f}%)."
        if not self.enough and self.pre_outliers:
            s += (f" 표준오차가 큰 주된 이유는 사전 기간에 크게 벗어난 시점이 있기 때문입니다: {self.outlier_text()}."
                  " 기준선이 흔들려 사후 기간을 늘리는 것만으로는 해결할 수 없습니다."
                  " 그 시점에 다른 정책·영업 중단·집계 변경이 있었는지 먼저 확인해야 합니다.")
            return s
        if not self.enough and self.post_reachable is False:
            s += (f" 정책 전 기간이 짧아(사전 {self.n_pre}{self.unit}) 사후 기간을 아무리 늘려도 기준 효과를 구분할 수 없습니다."
                  " 더 긴 정책 전 기간이나 더 세밀한 자료(일 단위·전체 카드사 등)가 필요합니다.")
            return s
        if not self.enough and self.post_needed:
            s += (f" 기준 효과를 구분하려면 표준오차가 지금의 1/{self.se_ratio:.1f} 이하여야 합니다. "
                  f"정책이 계속되는 경우 사후 기간 약 {self.post_needed}{self.unit}(현재 {self.n_post}{self.unit}), "
                  f"끝난 정책이면 더 세밀한 자료(일 단위·전체 카드사 등)가 필요합니다.")
        return s

    def needed_ko(self) -> str:
        if self.pre_outliers:
            return (f"사전 기간 이상 시점({', '.join(o['time'] for o in self.pre_outliers)})에 무슨 일이 있었는지 확인할 자료"
                    "(다른 정책·영업 중단·집계 기준 변경 기록). 원인이 확인되면 그 시점을 다루는 설계를 새로 등록(이 판정은 그대로 기록)")
        if self.post_reachable is False:
            return "더 긴 정책 전 기간의 자료 또는 더 세밀한 단위(일 단위·전체 카드사 등)의 자료(사후 기간만 늘려서는 해결 불가)"
        return (f"표준오차를 1/{self.se_ratio:.1f}로 줄일 자료: 사후 약 {self.post_needed}{self.unit}의 관측"
                f"(정책이 계속되는 경우) 또는 더 세밀한 단위의 자료")


def ratio_text(dev_log: float) -> str:
    """로그 차이를 '약 1/85 수준', '약 3.2배'처럼 읽을 수 있게."""
    r = float(np.exp(dev_log))
    if r < 1:
        return f"약 1/{1 / r:.0f} 수준" if r < 0.5 else f"약 {r * 100:.0f}% 수준"
    return f"약 {r:.1f}배"


def mde_text(mde_log: float) -> str:
    """구분 가능한 최소 효과. 로그 0.7(약 ±100%)을 넘으면 퍼센트 대칭 표기(±1702%)가 오해를 부르므로 배수로 쓴다."""
    if mde_log > 0.7:
        return f"약 {np.exp(mde_log):.1f}배 증가 또는 {-np.expm1(-mde_log) * 100:.0f}% 감소"
    return f"±{np.expm1(mde_log) * 100:.1f}%"


def mde_short(mde_log: float) -> str:
    """화면 타일용 짧은 표기."""
    if mde_log > 0.7:
        return f"×{np.exp(mde_log):.1f} / {-np.expm1(-mde_log) * 100:.0f}%↓"
    return f"±{np.expm1(mde_log) * 100:.1f}%"


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
           alpha: float = 0.05, power: float = 0.8, outliers: list | None = None, n_pre: int | None = None) -> Power:
    cands = [(se_analytic, "분석적 표준오차")]
    if se_placebo is not None and np.isfinite(se_placebo):
        cands.append((se_placebo, "위약 분포 표준편차"))
    se, src = max(cands, key=lambda c: c[0] if np.isfinite(c[0]) else -1)
    m = mde(se, alpha, power)
    ref_log = float(np.log1p(reference_pct))
    enough = bool(m <= ref_log)
    need, reach = None, None
    if not enough and n_post > 0 and ref_log > 0:
        r = (ref_log / m) ** 2                               # 분산을 이 비율로 줄여야 한다
        if n_pre:
            inv = r * (1 / n_pre + 1 / n_post) - 1 / n_pre   # 필요한 1/n_사후
            reach = bool(inv > 0)
            need = int(np.ceil(1 / inv)) if reach else None
        else:
            need = int(np.ceil(n_post / r))
    return Power(se, src, m, float(np.expm1(m)), reference_pct, enough, n_post, need, unit, list(outliers or []),
                 reach, n_pre)
