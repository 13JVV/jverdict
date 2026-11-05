"""데이터 품질 툴: 효과를 추정하기 전에 데이터를 의심한다.

- check_coverage: 전체·집단 결측, 부분 집계, 분류 신설·소멸, 단위 누락
- diagnose: 정책 시점의 측정 단절 진단(근거 E2~E5, 판정 상한)
- scan_breaks: 시점을 모르는 단절 탐색
- external_check: 공식 통계 대조
"""
from .breaks import Diagnosis, diagnose, scan_breaks
from .coverage import CoverageReport, check_coverage, exclusion_mask
from .external import ExternalStat, external_check
from ._panel import to_wide

__all__ = ["check_coverage", "exclusion_mask", "CoverageReport", "diagnose", "Diagnosis",
           "scan_breaks", "external_check", "ExternalStat", "to_wide"]
