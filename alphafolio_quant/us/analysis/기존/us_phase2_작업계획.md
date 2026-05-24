# US Quant System Phase 2 작업계획

**작성일**: 2025-11-26
**목표**: 정상 작동하는 퀀트 모델 구축 (단순 IC 개선이 아닌 근본적 문제 해결)

---

## 1. 현재 상태 요약

### 1.1 Phase 1 결과 (재설계 후)

| 지표 | 재설계 전 | 재설계 후 | 상태 |
|------|----------|----------|------|
| 전체 IC (10일) | -0.053 | **+0.081** | 개선 |
| Decile 승률 단조성 | 역전 | D1(51.9%)→D10(57.0%) | 개선 |
| Growth Factor IC | - | 0.048~0.086 | 우수 |
| **Momentum Factor IC** | 역작동 | -0.093~-0.019 | **미해결** |
| **매수 신호** | 역작동 | 승률 40.2%, 수익률 -1.2% | **미해결** |
| **Healthcare IC** | - | -0.068 | **미해결** |

### 1.2 데이터 현황 (2025-11-26 확인)

```
us_stock_basic 테이블:
- 전체: 4,723 종목
- 섹터 정상: 4,677개 (99%)
- 섹터 미분류: 46개 (0.97%)
  - NULL/Empty: 4개
  - NONE: 37개
  - OTHER: 5개

시가총액 필터: 제거 완료 (2025-11-26)
- 이전: market_cap > $500M → 2,435개만 분석
- 현재: 전체 종목 분석 가능
```

### 1.3 미해결 문제

| 순위 | 문제 | 영향도 | Deep Dive 필요 |
|------|------|--------|---------------|
| 1 | Momentum Factor 역작동 | 심각 | O |
| 2 | 매수 신호 역작동 (승률 40%) | 심각 | O |
| 3 | Healthcare 섹터 IC -0.068 | 높음 | O |
| 4 | NASDAQ IC -0.029 | 높음 | O |
| 5 | NONE 섹터 벤치마크 불안정 | 중간 | X (코드 수정) |

---

## 2. Phase 2 작업 구조

### 2.1 작업 흐름

```
[Phase 2-1] Deep Dive 분석 (원인 파악)
│
├── us_phase2_deep_dive.py 개발 및 실행
│   ├── STEP 1: Momentum Factor 전략별 분석 (M1~M20)
│   ├── STEP 2: Healthcare 섹터 상세 분석
│   └── STEP 3: Exchange (NYSE/NASDAQ) 분석
│
└── 분석 결과 보고서: us_phase2_deep_dive_결과.md

[Phase 2-2] 모델 결함 수정
│
└── us_sector_benchmarks.py 수정
    └── NONE/OTHER 섹터 fallback 로직 추가

[Phase 2-3] 팩터 재설계 (분석 결과 기반 결정)
│
├── Momentum Factor 개선 (분석 결과에 따라)
├── Healthcare 대응 로직 (분석 결과에 따라)
└── Phase 2 검증 분석 실행
```

---

## 3. Phase 2-1: Deep Dive 분석

### 3.1 통합 분석 파일

**파일명**: `us_phase2_deep_dive.py`
**위치**: `C:\project\alpha\quant\us\`

3개 분석을 하나의 파일로 통합하여 한 번 실행으로 완료:

```python
"""
US Quant Phase 2 - Deep Dive Analysis
=====================================

한 번 실행으로 3가지 분석 모두 수행:
- STEP 1: Momentum Factor Deep Dive (M1~M20)
- STEP 2: Healthcare Sector Deep Dive
- STEP 3: Exchange Deep Dive (NYSE vs NASDAQ)

Execute: python us/us_phase2_deep_dive.py
Output: C:\project\alpha\quant\us\result\ 폴더에 CSV 파일 생성
"""
```

### 3.2 STEP 1: Momentum Factor Deep Dive

**분석 항목:**

| 분석 | 내용 | 출력 파일 |
|------|------|----------|
| 전략별 IC | M1~M20 개별 전략의 기간별 IC | `us_phase2_momentum_strategies_ic.csv` |
| 섹터별 IC | 각 전략이 어느 섹터에서 작동/역작동 | `us_phase2_momentum_by_sector.csv` |
| Rolling IC | 날짜별 IC 안정성 | `us_phase2_momentum_rolling.csv` |
| Decile 테스트 | 전략별 점수-수익률 단조성 | `us_phase2_momentum_decile.csv` |
| 문제 종목 | 고점수-저수익 / 저점수-고수익 케이스 | `us_phase2_momentum_problem_stocks.csv` |

**핵심 질문:**
1. M1~M20 중 어떤 전략이 역작동하는가?
2. 가격 모멘텀 vs 기술적 지표 중 어느 쪽이 문제인가?
3. 특정 섹터에서만 역작동하는가?

### 3.3 STEP 2: Healthcare Sector Deep Dive

**분석 항목:**

| 분석 | 내용 | 출력 파일 |
|------|------|----------|
| 서브 업종별 IC | Biotechnology, Pharmaceuticals, Medical Devices 등 | `us_phase2_healthcare_by_industry.csv` |
| 팩터별 IC | Healthcare 내 Value/Quality/Momentum/Growth 효과 | `us_phase2_healthcare_by_factor.csv` |
| 시가총액별 IC | Large vs Small Cap Healthcare | `us_phase2_healthcare_by_marketcap.csv` |
| 문제 종목 | 고점수-급락 / 저점수-급등 케이스 | `us_phase2_healthcare_problem_stocks.csv` |

**핵심 질문:**
1. Biotechnology만 문제인가, Healthcare 전체가 문제인가?
2. Healthcare에서 어떤 팩터가 유효한가?
3. 대형 제약사 vs 소형 바이오텍 패턴 차이는?

### 3.4 STEP 3: Exchange Deep Dive

**분석 항목:**

| 분석 | 내용 | 출력 파일 |
|------|------|----------|
| 섹터 분포 | NYSE/NASDAQ 섹터 구성 비교 | `us_phase2_exchange_sector_dist.csv` |
| 팩터별 IC | 거래소별 팩터 효과 | `us_phase2_exchange_factor_ic.csv` |
| 교차 분석 | 섹터 × 거래소 IC 히트맵 | `us_phase2_exchange_cross_analysis.csv` |

**핵심 질문:**
1. NASDAQ의 음의 IC는 어느 섹터에서 발생하는가?
2. 같은 섹터라도 거래소에 따라 IC가 다른가?

---

## 4. Phase 2-2: 모델 결함 수정

### 4.1 NONE 섹터 처리

**문제**: sector='NONE'인 37개 종목은 벤치마크가 불안정 (37개 종목으로만 계산)

**해결**: us_sector_benchmarks.py에 fallback 로직 추가

```python
async def get_sector_benchmarks(self, sector: str) -> Dict:
    """섹터 벤치마크 조회 (NONE/OTHER는 전체 시장 평균 사용)"""

    if sector in ['NONE', 'OTHER', '', None]:
        return await self._get_market_average_benchmarks()

    # 기존 로직...
```

---

## 5. 분석 결과에 따른 다음 단계

Deep Dive 분석 결과에 따라 Phase 2-3 작업 결정:

| 분석 결과 | 대응 방향 |
|----------|----------|
| 특정 M전략만 역작동 | 해당 전략 재설계 또는 가중치 조정 |
| 가격 모멘텀 전체 역작동 | Earnings Revision 기반 모멘텀으로 대체 |
| Biotechnology만 문제 | Biotech 전용 팩터 또는 별도 모델 |
| Healthcare 전체 문제 | Healthcare 섹터 전용 점수 체계 |
| NASDAQ 특정 섹터 문제 | 거래소별 섹터 분리 처리 |

---

## 6. 실행 순서

```
1단계: Deep Dive 분석 파일 개발
   - us_phase2_deep_dive.py 생성
   - 참고: C:\project\alpha\quant\phase3_7_deep_dive.py (한국 모멘텀 분석)

2단계: 분석 실행
   - python us/us_phase2_deep_dive.py
   - 예상 시간: 1-2시간 (체크포인트 기능 포함)

3단계: 결과 분석 및 보고서 작성
   - CSV 파일 분석
   - us_phase2_deep_dive_결과.md 작성

4단계: 모델 결함 수정
   - us_sector_benchmarks.py NONE 섹터 fallback 추가

5단계: 팩터 재설계 (분석 결과 기반)
   - 구체적 작업은 분석 결과에 따라 결정

6단계: Phase 2 검증
   - 수정된 시스템으로 IC 분석 재실행
   - Phase 1 결과와 비교
```

---

## 7. 참고 파일

| 파일 | 용도 |
|------|------|
| `C:\project\alpha\quant\phase3_7_deep_dive.py` | 한국 모멘텀 Deep Dive (구조 참고) |
| `C:\project\alpha\quant\us\result\US_Quant_System_Diagnosis_Report.md` | 기존 진단 보고서 |
| `C:\project\alpha\quant\us\result\미국 phase1 분석결과.md` | Phase 1 분석 결과 |
| `C:\project\alpha\quant\us\us_main_v2.py` | 현재 퀀트 시스템 메인 |
| `C:\project\alpha\quant\us\us_momentum_factor_v2.py` | 현재 모멘텀 팩터 |
| `C:\project\alpha\quant\us\us_sector_benchmarks.py` | 섹터 벤치마크 |

---

## 8. 다음 세션 시작 방법

```
1. 이 파일 읽기:
   C:\project\alpha\quant\us\result\us_phase2_작업계획.md

2. Deep Dive 분석 파일 개발:
   us_phase2_deep_dive.py 생성
   (phase3_7_deep_dive.py 구조 참고)

3. 분석 실행 및 결과 확인

4. 분석 결과에 따른 다음 단계 결정
```

---

## 9. Phase 2-3 완료 내용 (2025-11-26)

### 9.1 EM3/EM4 Fundamental Momentum 대체

**파일:** `us_momentum_factor_v2.py`

| 전략 | 기존 (Price Momentum) | 변경 (Fundamental Momentum) |
|------|----------------------|----------------------------|
| **EM3** | Price vs MA (가격 이동평균 대비) | EPS Estimate Revision (30일 EPS 추정치 변화율) |
| **EM4** | Breakout Score (돌파 점수) | Revenue YoY Growth (매출 성장률) |

**Fallback 전략:**
```python
# EM3 Fallback
1. us_earnings_estimates.eps_estimate_average_30_days_ago → 변화율 계산
2. us_stock_basic.earnings_growth (YoY EPS 성장률)
3. 50점 (중립)

# EM4 Fallback
1. us_stock_basic.revenue_growth (YoY 매출 성장률)
2. 50점 (중립)
```

### 9.2 EM6 us_earnings_estimates 테이블 사용

**파일:** `us_momentum_factor_v2.py`

**기존:** us_earnings 테이블 (존재하지 않음) → 항상 실패

**변경:** us_earnings_estimates 테이블 사용
```python
Score = Net Revision Direction (60%) + EPS Change Rate (40%)

# Net Revision Direction
- revision_up_7d - revision_down_7d
- 범위: -100 ~ +100 → 0 ~ 100 정규화

# EPS Change Rate
- (eps_current - eps_30d_ago) / eps_30d_ago * 100
- 상하한 클리핑 후 정규화
```

### 9.3 NONE 섹터 Fallback 추가

**파일:** `us_sector_benchmarks.py`

**변경 내용:**
```python
async def get_sector_benchmarks(self, sector: str) -> Dict:
    # NONE/OTHER/empty 섹터 → 전체 시장 평균 사용
    if sector in ['NONE', 'OTHER', '', None] or not sector:
        return await self._get_market_average_benchmarks()
    # 기존 로직...
```

**시장 평균 벤치마크 항목:**
- Valuation: PE, Forward PE, PEG, PS, EV/EBITDA (median)
- Quality: Gross Margin, Operating Margin, ROE, ROA (median)
- Growth: Revenue Growth, EPS Growth (median)
- Momentum: 1M, 3M, 6M return (median)

---

## 10. 다음 단계: Phase 2 검증

### 10.1 검증 목표

| 지표 | Phase 1 (수정 전) | 목표 |
|------|------------------|------|
| Momentum IC (30일) | -0.056 | > +0.03 |
| Healthcare IC (30일) | -0.068 | > 0.00 |
| 전체 IC (30일) | +0.011 | > +0.05 |
| 매수 신호 승률 | 40.2% | > 52% |

### 10.2 검증 방법

```
1단계: us_main_v2.py 실행 (새 점수 계산)
2단계: IC 분석 스크립트 실행
3단계: Phase 1 결과와 비교
4단계: 검증 보고서 작성
```

---

## 11. 수정 이력

| 날짜 | 내용 |
|------|------|
| 2025-11-26 | us_main_v2.py 시가총액 $500M 필터 제거 |
| 2025-11-26 | Phase 2 작업계획 수립 |
| 2025-11-26 | Phase 2-1 Deep Dive 분석 완료 (us_phase2_deep_dive.py) |
| 2025-11-26 | Phase 2-2 NONE 섹터 fallback 추가 (us_sector_benchmarks.py) |
| 2025-11-26 | Phase 2-3 EM3/EM4 Fundamental Momentum 대체 (us_momentum_factor_v2.py) |
| 2025-11-26 | Phase 2-3 EM6 us_earnings_estimates 사용 (us_momentum_factor_v2.py) |

---

*작성: Claude Code*
*상태: Phase 2-3 완료, Phase 2 검증 대기*
