"""정책 카드 라이브러리, 공식 통계, 사건 달력.

- 정책 카드: 템플릿 모드(LLM 없음)에서 질문을 카드로 바꿀 때 쓰는 등록 사례. 온라인 모드에서도 사람 확인 화면의 기본값이 된다.
- 공식 통계: 외부 통계 대조(quality.external_check)에 쓰는 발표 수치. 출처를 함께 둔다.
- 사건 달력: Skeptic이 "같은 시기에 다른 일이 있었나"를 볼 때 쓰는 목록. 확인된 출처가 있는 것만 넣는다.
"""
from __future__ import annotations

from jverdict.quality.external import ExternalStat
from jverdict.synth.generator import LUNAR_HOLIDAYS

OFFICIAL_STATS: dict[str, ExternalStat] = {
    "youth_free_483": ExternalStat(
        "어린이·청소년 무료 이용 누적 483만 건(도 발표, 2025-08~2026-02)", 4_830_000, "2025-08-01", "2026-02-28",
        ["청소년", "어린이"], "제주도 2026-04-05 발표(헤드라인제주·제주일보 보도)", same_definition=False),
}

NEEDED_A = ["교통복지카드(무료) 승차 기록이 포함된 이용자 유형별 승차 자료",
            "2025-08-01 전후 이용자 유형 분류 기준 변경 이력(데이터 제공처 확인)"]

# 사건 달력: start, end, name, affects(영향받을 수 있는 집단·업종 설명), source, added(달력에 넣은 날, 없으면 처음부터)
EVENTS: list[dict] = [
    {"start": "2024-09-16", "end": "2024-11-24", "name": "청소년 문화축제 연계 무료 셔틀·공연(합성 시연 사례 F)",
     "affects": "청소년 이동", "source": "가상 행사(합성데이터 시연)", "dataset": "synthetic_bus_event", "added": "2026-11-07"},
    {"start": "2020-02-18", "end": "2020-12-31", "name": "코로나19 1차·2차 확산", "affects": "전체 이동·소비",
     "source": "공지 사실"},
    {"start": "2020-03-01", "end": "2020-05-31", "name": "등교 연기·온라인 개학(전국)", "affects": "청소년·어린이 이동",
     "source": "교육부 발표(2020)"},
    {"start": "2020-04-30", "end": "2020-05-05", "name": "2020 황금연휴 국내 관광 반등", "affects": "관광객 지출, 음식·숙박 업종",
     "source": "findings F5"},
    {"start": "2020-03-01", "end": "2020-05-31", "name": "국제유가 급락", "affects": "주유소·가스충전 업종 금액",
     "source": "findings F5"},
    {"start": "2021-09-06", "end": "2021-12-31", "name": "상생 국민지원금 지급·사용", "affects": "도민 소비(소득 하위 88%)",
     "source": "relief2020.py 위약 제외 기간"},
    {"start": "2022-08-12", "end": "2022-11-23", "name": "버스 정류소 파일 전체 결측(데이터)", "affects": "버스 자료 전체",
     "source": "findings F7", "dataset": "bus_station"},
    {"start": "2022-03-17", "end": "2022-05-31", "name": "오미크론 유행 정점(3-17 신규 확진 약 62만 명) 뒤 감소기",
     "affects": "대면 소비 전반(주점·음식점 등). 확진 감소에 따른 자연 회복", "source": "질병관리청 발표 보도(2022-03-17~23)", "added": "2026-11-06"},
    {"start": "2022-04-18", "end": "2022-04-18", "name": "사회적 거리두기 전면 해제", "affects": "주점·음식점 영업시간, 사적모임",
     "source": "중앙재난안전대책본부 발표(2022-04-15)", "case": "E", "added": "2026-11-06"},
    {"start": "2022-04-25", "end": "2022-04-25", "name": "실내 취식 금지 해제(거리두기 해제의 마지막 단계)", "affects": "실내 취식 시설",
     "source": "중앙재난안전대책본부 발표(2022-04-15)", "case": "E", "added": "2026-11-06"},
    {"start": "2023-01-02", "end": "2023-02-10", "name": "한국의 중국인 단기비자 발급 중단", "affects": "중국인 방문",
     "source": "사례 D 사전 등록 3장"},
    {"start": "2023-05-01", "end": "2023-06-30", "name": "카드 파일 외국인 4개 집단 결측(데이터)", "affects": "중국·일본·동남아·기타외국",
     "source": "findings F6", "dataset": "card_sector"},
    {"start": "2023-08-10", "end": "2023-08-10", "name": "중국 단체관광 허용", "affects": "중국인 방문", "source": "사례 D 사전 등록", "case": "D"},
    {"start": "2023-08-31", "end": "2023-09-30", "name": "크루즈 재입항(전후)", "affects": "중국인 방문·면세점", "source": "사례 D 사전 등록 9장"},
    {"start": "2025-01-01", "end": "2025-01-01", "name": "버스 이용자 유형 '기타' 신설(데이터)", "affects": "어린이 집계",
     "source": "findings F7", "dataset": "bus_station"},
    {"start": "2025-08-01", "end": "2025-08-01", "name": "청소년 버스 무료·어린이 무료 확대", "affects": "청소년·어린이",
     "source": "제주도 발표", "case": "A"},
] + [{"start": h, "end": h, "name": "설·추석 연휴", "affects": "전체 이동(연휴 3일)", "source": "양력 환산 명절표"}
     for h in LUNAR_HOLIDAYS]

VACATIONS = {"여름방학(대략)": ("07-20", "08-20"), "겨울방학(대략)": ("12-28", "02-25")}


def _bus_series(*names):
    return [{"name": n, "groups": [n], "units_tier": None, "units_list": []} for n in names]


CARDS: dict[str, dict] = {
    "A": {
        "card": {"policy": "청소년 버스 무료(어린이 무료 확대)", "question": "청소년 버스 무료화(2025-08-01)로 청소년의 버스 이용이 늘었는가?",
                 "start_date": "2025-08-01", "end_date": None, "treated": "청소년(만 13~18세), 어린이 무료 확대 동시 시행",
                 "outcome": "이용자 유형별 일 단위 버스 이용 인원", "expected": "increase", "region": "제주도 전역",
                 "source": "제주도 발표", "reference_effect": 0.10, "needed_hint": NEEDED_A, "prereg": None},
        "keywords": ["청소년", "버스", "무료"], "must": ["무료"],
        "design": {
            "dataset": "bus_station", "method": "its",
            "series": _bus_series("청소년", "어린이", "일반", "경로", "장애 일반"),
            "target": "청소년", "reference": ["일반"],
            "placebo": [{"target": "경로", "reference": ["일반"]}, {"target": "장애 일반", "reference": ["일반"]}],
            "alt_references": [], "pre": None, "post": None, "season_years": 3, "exclude": [],
            "reference_effect": 0.10, "expected": "increase", "official_stats": ["youth_free_483"], "panel": None,
            "choices": [
                {"decision": "대조 집단: 일반", "reason": "정책 대상이 아니고 규모가 가장 커 날씨·관광 등 공통 변화를 가장 잘 반영"},
                {"decision": "공간 위약: 경로·장애 일반", "reason": "정책 대상이 아닌 다른 할인 유형. 같은 때 변하면 대조 쪽 사건을 의심"},
                {"decision": "계절 차분 3년", "reason": "시행일(8월)이 방학과 겹쳐 같은 요일·같은 시기 과거 값과 비교"},
                {"decision": "공식 통계 대조: 도 발표 483만 건", "reason": "공개 데이터 집계와 공식 실적의 규모 확인"}]},
    },
    "B": {
        "card": {"policy": "1차 긴급재난지원금", "question": "1차 긴급재난지원금(2020-05)이 사용 가능 업종의 제주도민 카드 소비를 늘렸는가?",
                 "start_date": "2020-05-01", "end_date": "2020-08-31", "treated": "제주도민(지원금 수령), 사용 가능 업종",
                 "outcome": "업종별 월 카드 이용 금액", "expected": "increase", "region": "제주도",
                 "source": "행정안전부·제주도 발표(2020)", "reference_effect": 0.10, "needed_hint": [],
                 "prereg": "docs/prereg/case_relief2020_engine.md"},
        "keywords": ["재난지원금", "지원금", "카드"], "must": ["지원금"],
        "design": {
            "dataset": "card_sector", "method": "its",
            "series": [
                {"name": "도민_가능", "groups": ["제주도민"], "units_tier": "tier:가능", "units_list": []},
                {"name": "도민_불가", "groups": ["제주도민"], "units_tier": "tier:불가", "units_list": []},
                {"name": "도민_대형종합", "groups": ["제주도민"], "units_tier": None, "units_list": ["기타 대형 종합 소매업"]},
                {"name": "도민_불가면세", "groups": ["제주도민"], "units_tier": None,
                 "units_list": ["기타 대형 종합 소매업", "일반유흥 주점업", "면세점"]},
                {"name": "법인_가능", "groups": ["법인"], "units_tier": "tier:가능", "units_list": []},
                {"name": "법인_불가", "groups": ["법인"], "units_tier": "tier:불가", "units_list": []}],
            "target": "도민_가능", "reference": ["도민_불가"],
            "placebo": [{"target": "법인_가능", "reference": ["법인_불가"]}],
            "alt_references": [{"label": "대형종합만", "reference": ["도민_대형종합"]},
                               {"label": "대형종합+유흥+면세", "reference": ["도민_불가면세"]}],
            "pre": 3, "post": 4, "season_years": 1,
            "exclude": [{"start": "2021-09-01", "end": "2021-12-01", "reason": "상생 국민지원금"},
                        {"start": "2022-09-01", "end": "2022-12-01", "reason": "상생 국민지원금의 1년 뒤 메아리"}],
            "reference_effect": 0.10, "expected": None, "official_stats": [], "panel": None,
            "choices": [{"decision": "사전 등록 문서 사용", "reason": "docs/prereg/case_relief2020_engine.md (잠금 해시 확인)"}]},
    },
    "D": {
        "card": {"policy": "중국 단체관광 재개", "question": "중국 단체관광 재개(2023-08-10) 이후 제주의 중국인 카드 소비가 다른 외국인 대비 늘었는가?",
                 "start_date": "2023-08-01", "end_date": None, "treated": "중국인 방문객",
                 "outcome": "업종별 월 카드 이용 금액(2019 같은 달 대비)", "expected": "increase", "region": "제주도",
                 "source": "중국 문화여유부 발표(2023-08-10)", "reference_effect": 0.20, "needed_hint": [],
                 "prereg": "docs/prereg/case_china_group_tour_2023.md"},
        "keywords": ["중국", "단체관광", "유커", "관광"], "must": ["중국", "유커"],
        "design": {
            "dataset": "card_sector", "method": "did_panel", "series": [], "target": "중국", "reference": [],
            "placebo": [], "alt_references": [], "pre": None, "post": None, "season_years": 1, "exclude": [],
            "reference_effect": 0.20, "expected": "increase", "official_stats": [],
            "panel": {"groups": ["중국", "일본", "동남아", "기타외국"], "treated": "중국",
                      "months": ["2023-03-01", "2023-04-01", "2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01",
                                 "2023-11-01", "2023-12-01", "2024-01-01"],
                      "base_year": 2019, "sector_month_fe": True, "ref": "2023-07-01",
                      "sens": [{"label": "S1 사후 2023-09부터(8월 제외)", "drop_months": ["2023-08-01"], "drop_biz": [], "weighted": False},
                               {"label": "S2 2019 연간 금액 가중", "drop_months": [], "drop_biz": [], "weighted": True},
                               {"label": "S3 면세점 제외", "drop_months": [], "drop_biz": ["면세점"], "weighted": False}]},
            "choices": [{"decision": "사전 등록 문서 사용(수정 1)", "reason": "docs/prereg/case_china_group_tour_2023.md (잠금 해시 확인)"}]},
    },
    "E": {
        "card": {"policy": "사회적 거리두기 전면 해제", "question": "사회적 거리두기 전면 해제(2022-04-18)로 제주도민의 주점 소비가 늘었는가?",
                 "start_date": "2022-04-01", "end_date": None,
                 "treated": "영업시간·모임 제한을 받던 주점(일반유흥 주점업, 기타 주점업). 시행 4-18이라 4월은 절반만 처치(월 단위)",
                 "outcome": "업종별 월 카드 이용 금액(제주도민)", "expected": "increase", "region": "제주도(전국 동시 시행)",
                 "source": "중앙재난안전대책본부 발표(2022-04-15)", "reference_effect": 0.10, "needed_hint": [],
                 "prereg": "docs/prereg/case_distancing_2022.md"},
        "keywords": ["거리두기", "해제", "주점", "영업시간", "방역"], "must": ["거리두기"],
        "design": {
            "dataset": "card_sector", "method": "its", "season": "none",
            "series": [
                {"name": "도민_주점", "groups": ["제주도민"], "units_tier": None, "units_list": ["일반유흥 주점업", "기타 주점업"]},
                {"name": "도민_식료품", "groups": ["제주도민"], "units_tier": None,
                 "units_list": ["슈퍼마켓", "기타음식료품위주종합소매업", "과실 및 채소 소매업", "수산물 소매업", "육류 소매업"]},
                {"name": "도민_대형종합", "groups": ["제주도민"], "units_tier": None, "units_list": ["기타 대형 종합 소매업"]},
                {"name": "도민_편의점", "groups": ["제주도민"], "units_tier": None, "units_list": ["체인화 편의점"]},
                {"name": "도민_화장품건강", "groups": ["제주도민"], "units_tier": None,
                 "units_list": ["화장품 및 방향제 소매업", "건강보조식품 소매업"]},
                {"name": "도민_잡화", "groups": ["제주도민"], "units_tier": None, "units_list": ["그외 기타 종합 소매업"]}],
            "target": "도민_주점", "reference": ["도민_식료품"],
            "placebo": [{"target": "도민_화장품건강", "reference": ["도민_식료품"]},
                        {"target": "도민_잡화", "reference": ["도민_식료품"]}],
            "alt_references": [{"label": "대형종합만", "reference": ["도민_대형종합"]},
                               {"label": "편의점만", "reference": ["도민_편의점"]}],
            "pre": 12, "post": 6, "season_years": 1,
            "exclude": [{"start": "2020-02-01", "end": "2022-04-30",
                         "reason": "코로나19 방역 기간: 같은 종류의 조치(영업시간·모임 제한)가 계속 바뀌어 '정책 없는' 가짜 시행일이 될 수 없음"}],
            "reference_effect": 0.10, "expected": "increase", "official_stats": [], "panel": None,
            "choices": [{"decision": "사전 등록 문서 사용", "reason": "docs/prereg/case_distancing_2022.md (잠금 해시 확인)"}]},
    },
    "C": {
        "card": {"policy": "(시연) 청소년 버스 요금 할인 — 합성데이터", "question": "(시연) 2024-03-04 청소년 요금 할인으로 청소년 버스 이용이 늘었는가?",
                 "start_date": "2024-03-04", "end_date": None, "treated": "청소년", "outcome": "이용자 유형별 일 단위 버스 이용 인원",
                 "expected": "increase", "region": "가상", "source": "합성데이터(정답 공개는 판정 뒤)", "reference_effect": 0.10,
                 "needed_hint": [], "prereg": None},
        "keywords": ["시연", "합성", "가상"], "must": ["시연", "합성"],
        "design": {
            "dataset": "synthetic_bus", "method": "its", "series": _bus_series("청소년", "어린이", "일반", "경로"),
            "target": "청소년", "reference": ["일반"],
            "placebo": [{"target": "경로", "reference": ["일반"]}, {"target": "어린이", "reference": ["일반"]}],
            "alt_references": [], "pre": None, "post": None, "season_years": 3, "exclude": [], "reference_effect": 0.10,
            "expected": "increase", "official_stats": [], "panel": None,
            "choices": [{"decision": "대조 집단: 일반", "reason": "정책 대상이 아니고 규모가 가장 큼"},
                        {"decision": "공간 위약: 경로·어린이", "reason": "정책 대상이 아닌 집단"}]},
    },
    "F": {
        "card": {"policy": "(시연) 청소년 야간 버스 증편 — 합성데이터 2",
                 "question": "(시연) 2024-09-02 청소년 야간 버스 증편으로 청소년 버스 이용이 늘었는가?",
                 "start_date": "2024-09-02", "end_date": None, "treated": "청소년", "outcome": "이용자 유형별 일 단위 버스 이용 인원",
                 "expected": "increase", "region": "가상", "source": "합성데이터 2(정답 공개는 판정 뒤)", "reference_effect": 0.10,
                 "needed_hint": [], "prereg": None},
        "keywords": ["시연", "합성", "증편", "야간"], "must": ["증편"],
        "design": {
            "dataset": "synthetic_bus_event", "method": "its", "series": _bus_series("청소년", "어린이", "일반", "경로"),
            "target": "청소년", "reference": ["일반"],
            "placebo": [{"target": "경로", "reference": ["일반"]}, {"target": "어린이", "reference": ["일반"]}],
            "alt_references": [], "pre": None, "post": None, "season_years": 3, "exclude": [], "reference_effect": 0.10,
            "expected": "increase", "official_stats": [], "panel": None,
            "choices": [{"decision": "대조 집단: 일반", "reason": "정책 대상이 아니고 규모가 가장 큼"},
                        {"decision": "공간 위약: 경로·어린이", "reason": "정책 대상이 아닌 집단"}]},
    },
}


def match_card(question: str) -> str | None:
    """템플릿 모드의 정책 카드 찾기: 사례별 필수 핵심어가 하나 이상 있고, 핵심어가 가장 많이 맞는 사례."""
    q = (question or "").replace(" ", "")
    best, score = None, 0
    for k, c in CARDS.items():
        if not any(w in q for w in c["must"]):
            continue
        hit = sum(1 for w in c["keywords"] if w in q) + (5 if k in ("C", "F") else 0)   # 시연 표시는 다른 사례보다 우선
        if hit > score:
            best, score = k, hit
    return best


def events_between(start: str, end: str, dataset: str | None = None, as_of: str | None = None) -> list[dict]:
    """기간 안의 사건. 데이터 관련 사건은 그 데이터셋일 때만, 합성데이터에는 방학·명절만(생성기가 모형화한 것) 준다.

    as_of(YYYY-MM-DD): 그날까지 달력에 있던 사건만. 재실행은 원래 실행 날짜의 달력을 써야 같은 입력이 된다.
    'added'가 없는 사건은 처음부터 있던 것이다.
    """
    import pandas as pd
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    synth = dataset in ("synthetic_bus", "synthetic_bus_event")
    out = [ev for ev in EVENTS if pd.Timestamp(ev["start"]) <= e and pd.Timestamp(ev["end"]) >= s
           and (as_of is None or ev.get("added", "") <= as_of)
           and (ev.get("dataset") is None or ev.get("dataset") == dataset)
           and (not synth or ev["name"] == "설·추석 연휴" or ev.get("dataset") == dataset)]
    for y in range(s.year - 1, e.year + 1):
        for name, (a, b) in VACATIONS.items():
            vs = pd.Timestamp(f"{y}-{a}")
            ve = pd.Timestamp(f"{y + (1 if b < a else 0)}-{b}")
            if vs <= e and ve >= s:
                out.append({"start": str(vs.date()), "end": str(ve.date()), "name": name, "affects": "청소년·어린이 이동",
                            "source": "일반 학사 일정(학교마다 다름)"})
    return out
