# Best 5 Stock 선정 - AI 분석 가이드

이 문서는 `python scripts/daily_alpha.py` 실행 후 생성된 리포트(`docs/result/YYYY-MM-DD_alpha.md`)를
기반으로 Claude가 Step 7~8을 수행하기 위한 지식 베이스이다.

## 사전 조건

- `python scripts/daily_alpha.py` 실행 완료 (Step 1~6, AI 불필요)
- `docs/result/YYYY-MM-DD_alpha.md` 리포트 생성 확인
- 리포트의 "4. 투표 집계" 섹션에 후보 종목 리스트 존재

## Step 7: 기초 데이터 분석 (부실 종목 제외)

### 데이터 수집 방법

투표 집계의 모든 종목에 대해 다음 DB 쿼리로 데이터 수집:

```sql
-- 1. 종목 기본정보
SELECT code, name, market_cap, per, pbr, roe, debt_ratio
FROM stocks WHERE code IN (종목코드들);

-- 2. 최근 3년 연간 재무 (Q4 = 연간 누적)
SELECT stock_code, fiscal_year, revenue, operating_income, net_income,
       total_equity, roe, debt_ratio
FROM fundamental_history
WHERE stock_code IN (종목코드들) AND fiscal_quarter = 4 AND fiscal_year >= (올해-3);

-- 3. 최근 분기 재무 (최신 상태 확인)
SELECT stock_code, fiscal_year, fiscal_quarter, revenue, operating_income,
       net_income, roe, debt_ratio
FROM fundamental_history
WHERE stock_code IN (종목코드들)
ORDER BY fiscal_year DESC, fiscal_quarter DESC;

-- 4. 주가 추이 (재무 데이터 없는 종목용)
SELECT date, close, volume FROM daily_prices
WHERE stock_code = ? AND date >= '2년전날짜' ORDER BY date;
```

### 제외 규칙 (하나라도 해당시 제외)

| # | 규칙 | 기준 | 확인 방법 |
|---|------|------|----------|
| 1 | 연속 영업적자 | 2년 이상 연속 영업적자 | fundamental_history에서 Q4 operating_income < 0 연속 |
| 2 | 자본잠식 위험 | ROE < -20% 또는 부채비율 > 500% | stocks 테이블 또는 최근 분기 재무 |
| 3 | 자본 급감 | 최근 3년간 자본총계 50% 이상 감소 | total_equity 추이 비교 |
| 4 | 초소형주 + 재무 미비 | 시총 500억 미만 + fundamental_history 데이터 없음 | stocks.market_cap + 재무 존재 여부 |
| 5 | 극단적 고평가 + 이익 악화 | PER > 100 + 순이익 전년 대비 50% 이상 감소 | stocks.per + net_income 추이 |
| 6 | 주가 급락 + 펀더멘털 악화 | 2년간 -50% 이상 하락 + ROE 음수 | daily_prices 추이 + ROE |
| 7 | 우선주 유동성 부족 | 종목코드 끝자리 5/7/8/9(우선주) + 일평균 거래량 < 5만주 | 코드 패턴 + volume 평균 |

### 제외 판단 시 주의사항

- 일시적 적자 vs 구조적 적자 구분: 반도체(SK하이닉스 등)는 사이클 산업이므로 1년 적자만으로 제외하지 않음
- 부채비율은 업종별 차이 고려: 건설/중공업은 200~300%도 정상, 제조업은 150% 이상 주의
- 시총 기준은 절대적이지 않음: 재무 데이터가 있고 건전하면 소형주도 잔류 가능

## Step 8: 최종 Top 5 선정

### 평가 항목 및 가중치

| 항목 | 가중치 | 평가 기준 |
|------|--------|----------|
| 득표수 | 높음 | 3표 이상 우선. 알파 시그널이 많을수록 신뢰도 높음 |
| ROE | 높음 | 10% 이상 우수, 5~10% 양호, 5% 미만 보통 |
| 이익 성장성 | 높음 | 최근 3년 영업이익/순이익 증가 추세 |
| 매출 성장성 | 중간 | 최근 3년 매출 증가율 |
| 부채비율 개선 | 중간 | 부채비율 감소 추세 긍정적 |
| 유동성 | 중간 | 일평균 거래량 10만주 이상 선호, 1만주 미만 감점 |
| 밸류에이션 | 중간 | PER 20 이하 + PBR 1 이하 = 가치주, 합리적 범위 |
| 산업 트렌드 | 낮음 | 시장 테마/정책 수혜 여부 (참고 수준) |

### 산업/테마 판단 참고 (2025~2026 기준)

아래는 참고용이며, 시간이 지나면 변할 수 있으므로 최신 뉴스 확인 권장.

| 테마 | 수혜 업종/종목 예시 | 비고 |
|------|-------------------|------|
| AI/반도체 | SK하이닉스, 삼성전자, SK스퀘어 | HBM, AI 가속기 수요 |
| 전력인프라 | 효성중공업, HD현대일렉트릭, LS ELECTRIC | 데이터센터 전력 수요 급증 |
| 자동차/SDV | 현대차, 현대오토에버, 현대모비스 | 전기차 + 소프트웨어 전환 |
| 방산 | 한화에어로스페이스, LIG넥스원 | 글로벌 방산 수출 확대 |
| 바이오 | 삼성바이오로직스, 셀트리온 | CDMO 성장 |
| 조선 | HD한국조선해양, 한화오션 | LNG선 수주 호조 |

### 선정 시 균형 원칙

- **업종 분산**: 같은 업종에서 최대 2종목까지 (예: 반도체에서 SK하이닉스+SK스퀘어는 실질적으로 같은 노출)
- **시총 분산**: 대형주 위주 + 중소형 1~2개 혼합 가능
- **득표수 존중**: 4표 종목이 3표 종목보다 낮은 순위가 되려면 명확한 재무적 근거 필요

### 잔류 종목 중 탈락 처리

Top 5에 들지 못한 잔류 종목에 대해서도 탈락 사유를 간단히 기록:
- 예: "ROE 2.5% 저조", "성장 모멘텀 약함", "소형주 리스크" 등

## AI 프롬프트 템플릿

스크립트 실행 후 Claude에게 순서대로 요청:

### 프롬프트 1: 부실 종목 제외

```
오늘 리포트(docs/result/YYYY-MM-DD_alpha.md)의 투표 종목들의 최근 3년 기초적 데이터 분석을 통해서
나쁜 종목들은 빼고 나머지 종목들을 정리해줘. 뺀 종목들의 이유도 알려줘.
이 내용을 결과 파일에 같이 넣어줘.
```

### 프롬프트 2: 최종 5종목 선정

```
잔류 종목 중에서 최종 5개만 골라줘
```

### 프롬프트 3: 저장 및 커밋

```
이것도 결과 파일에 추가하고 커밋 push해줘
```

## 참조 파일

| 파일 | 용도 |
|------|------|
| `scripts/daily_alpha.py` | Step 1~6 자동 실행 |
| `docs/result/YYYY-MM-DD_alpha.md` | 일일 리포트 (Step 1~8 결과 누적) |
| `docs/best5stock.md` | 전체 프로세스 요약 |
| `data/lasps.db` | SQLite DB (daily_prices, stocks, fundamental_history) |
