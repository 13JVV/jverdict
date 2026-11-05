# 데이터 받는 방법

이 저장소에는 제주데이터허브 원자료와 가공 파일을 싣지 않았습니다(허브의 재배포 조건을 확인하지 못함).
아래 파일을 제주데이터허브(https://www.jejudatahub.net, 로그인 후 내려받기)에서 받아 `data/raw/`에 두면
모든 사례를 이 PC에서 다시 판정할 수 있습니다.

| 파일 | 허브 데이터명 |
|---|---|
| 이용자유형별버스정류소이용인원현황.csv | 이용자 유형별 버스정류소 이용 인원 현황 |
| 일별버스승차인원수현황.csv | 일별 버스 승차 인원 수 현황 |
| 일별시간대별버스승차인원수현황.csv | 일별 시간대별 버스 승차 인원 수 현황 |
| 월별버스이용자유형별패널티현황.csv | 월별 버스 이용자 유형별 패널티 현황 |
| 업종별성별카드이용정보.csv | 업종별 성별 카드 이용 정보 |

받은 뒤:

```bash
python -m jverdict.data.ingest_hub      # data/processed/*.parquet 만들기
python -m jverdict.data.sources         # 지문 기록(_sources.json)과 대조
python -m jverdict.doctor               # 사례를 새로 판정해 기록과 같은지 확인
```

`data/processed/_sources.json`에는 원자료와 가공 파일의 SHA-256 지문이 있습니다. 같은 파일을 받았는지 확인할 수 있습니다.
