# Pattern Finder - Claude Code 가이드

## 명령어 실행 규칙

모든 명령어는 반드시 `docker compose`를 통해 실행한다. 로컬 환경에서 직접 pip, pytest 등을 실행하지 않는다.

```bash
# 서비스 시작 (빌드 포함)
docker compose up -d --build backtester

# 명령어 실행 (실행 중인 컨테이너에서)
docker compose exec backtester pytest -v
docker compose exec backtester python -c "..."

# 서비스 중지
docker compose down
```

## 프로젝트 구조

- `backtester/` — 하나의 앱. 하나의 Dockerfile. 헥사고날 아키텍처(Ports & Adapters).
  - 각 도메인 패키지는 `domain/`(models, ports)과 `adapters/`(구현체)로 분리.
  - `data/` — 시장 데이터 수집/저장
    - `domain/ports.py`: `MarketDataPort` — OHLCV 데이터 조회 인터페이스
    - `adapters/yfinance_adapter.py`: yfinance 구현체. `end`는 inclusive하게 동작(yfinance 원래 exclusive)
    - `adapters/cached_market_data.py`: 파퀘 기반 디스크 캐시 데코레이터. `end >= today` 요청은 **자동 우회** — 장중 intraday 업데이트 반영
    - `adapters/wikipedia_universe.py`: S&P 500 / Nasdaq-100 종목 리스트
    - `domain/ports.py: IpoCalendarPort` — 기간별 IPO(첫 거래일·공모가). `adapters/finnhub_ipo_calendar.py`(Finnhub, 2010~, Mongo 월 캐시) → `adapters/alphavantage_listing_ipos.py`(LISTING_STATUS ipoDate 폴백). composition: `adapters/composed_ipo_calendar.py`
  - `pattern/` — 패턴 감지
    - `domain/ports.py`: `PatternDetector` — 패턴 감지 인터페이스 (EMA, 평균 거래량 헬퍼 포함)
    - `adapters/wedge_pop.py`, `adapters/exhaustion_extension_top.py` 등 — 개별 패턴 구현체
    - `helpers/pivots.py` — swing high/low, trendline fit (numpy 벡터화 완료)
  - `strategy/` — 전략 실행 (패턴 + 사이즈 + 엑싯 조합)
    - `adapters/wedgepop_strategy.py`: 단일 ticker Wedge Pop. 엑싯 규칙 5종(trendline_break, resistance_break, exhaustion_exit, breakeven_stop, smart_trail). No-leverage 포지션 캡
    - `adapters/multi_wedgepop_strategy.py`: 전체 universe에서 일자별 buy/sell ratio 최상위 종목 선택
  - `backtest/` — 백테스팅 엔진
  - `visualization/` — 시각화 (`adapters/plotly_charts.py`)
  - `signals/` — **매수 신호 watchlist + 영속화**
    - `domain/models.py`: `BuySignal`, `SignalStatus`(pending/taken/rejected/expired)
    - `domain/ports.py`: `SignalRepositoryPort`, `SignalScannerPort`
    - `adapters/in_memory_repo.py`: 프로세스 메모리 저장 (DB 불가 시 fallback)
    - `adapters/postgres_signal_repo.py`: **SQLAlchemy 기반** Postgres 어댑터
    - `adapters/orm.py`: `BuySignalRow` SQLAlchemy 모델 + 도메인 변환기
    - `adapters/universe_scanner.py`: 현재 필터로 최근 N일 신호 스캔. `refresh_targets()` / `build_signal_at(ticker, date)` 제공
  - `db/` — **DB 인프라** (공용)
    - `base.py`: `DeclarativeBase` (모든 ORM 모델이 상속)
    - `session.py`: engine singleton + `session_scope()`
  - `alembic/` — 마이그레이션 (컨테이너 기동 시 `alembic upgrade head` 자동 실행)
    - `env.py`: `DATABASE_URL` 환경변수 우선 사용, `target_metadata = Base.metadata`
    - `versions/`: 마이그레이션 파일
  - `streamlit_app.py` + `pages/` — Streamlit UI
    - `pages/1_Pattern_Detection.py`, `pages/2_Backtest_Results.py`, `pages/strategy.py`
    - `pages/3_Multi_Wedgepop.py`: universe 전체 백테스트
    - `pages/51_Lifecycle_Trade.py`: **«The Lifecycle Trade» IPO 생애주기** — 7장 스크린(IPO Alert/Rare Jewels/Liquidity Matters) + 6장 통계 재현 + 4매수지점×5매도규칙 $100K 시뮬 + ⑤ 계좌 시뮬(책 8장 자금관리로 그대로 돌리면 얼마). 엔진 `strategy/adapters/lifecycle_trade.py`, 계좌 엔진 `strategy/adapters/lifecycle_portfolio.py`, 배치 `lifecycle_batch_cli.py` → `lifecycle_portfolio_cli.py`, 문서 `docs/strategy_notes/라이프사이클_트레이드_*.md`
    - `pages/52_TQQQ_Crash_Switch.py`: **"TQQQ 폭락 스위치 18" — 2026-10-19 확정 전략의 운용 화면, 손잡이 없음.** 규칙은 `deep_correction_lab.CRASH_SWITCH_PARAMS/RULE` 에 이름 붙여 고정(200일선 & 고점 −18% 이탈, 20일선 재진입, 다음날 체결, TQQQ 100%). 오늘 보유 여부·트리거, 세후 성적표, 자산 곡선, 매도→재매수 이력. 선정 근거 `docs/strategy_notes/study_cards/README.md`
    - `pages/53_Evidence_Checklist.py`: **근거중심매매(온휘) 체크리스트 — 과거 시점 블라인드 판정 도구.** 티커·기준일을 넣으면 기준일까지의 주봉·일봉(·4시간봉)으로 이동평균선·RSI·캔들·FVG·매물대는 자동, 채널·피보나치는 스윙 피벗 근사치를 차트에 그려 사람이 확정, 5대 매매 근거(≥4/5)와 근거가중치(≥4점)로 베팅/단기/보류/매매 안 함 판정 → 기록 → "봉인 해제"로 이후 성과. 엔진 `strategy/adapters/evidence_checklist.py`(순수 pandas), 설계서 `docs/strategy_notes/근거중심매매_검증_체크리스트_2026-10.md`(사본 `pages/_shared/evidence_manual.md` — 문서 수정 시 같이 복사), 테스트 `tests/test_evidence_checklist.py`
    - `pages/54_Evidence_Screener.py`: **근거중심매매 스크리너** — 유니버스(nasdaq100/sp500/nasdaq_full/nasdaq_full@기준일/직접 입력, 전 종목은 `fill_recent_days` 로 캐시 선채움 + 최소 주가 필터) 전 종목에 53 의 자동 판정을 돌려 주봉 5대 근거 '예' ≥ N·가중치·4분면 조건을 통과한 후보만 표시, 종목을 고르면 53 페이지로 넘어감(세션 prefill). 기본 데이터 소스는 `data/adapters/yfinance_bulk.download_many`(200종목/요청, 나스닥 전 종목 약 3분, 캐시 미기록), 옵션으로 캐시 스택. 스캔 결과는 `st.cache_data` 6시간 + 세션, 통과 조건은 `apply_filter` 로 재스캔 없이 즉시 재적용. 엔진 `strategy/adapters/evidence_screen.py` (`run_screen(prices=...)`, `compute_rows`, `apply_filter`)
    - `pages/49_TQQQ_Deep_Correction.py`: 기존 ⑩ 급락 회피 **연구용** 그대로(격자·분포 탐색)
    - `pages/46_Weinstein_Screener.py`: 와인스타인 스크리너. 데이터 소스 라디오 기본 **yfinance 일괄**(`run_scan(prices=...)`, sp500 약 42초 다운로드 + 11초 스캔; 캐시 스택은 종목당 조회라 수 분). `weinstein_stage.classify_stages` 는 numpy 루프(행 단위 `.iat` 제거로 28초 → 2초)
    - `pages/4_Multi_Wedgepop_Signals.py`: **Multi Wedgepop 실매수 후보 + watchlist 관리** — `3_Multi_Wedgepop` 백테스트와 동일 설정으로 현재 시점 signal 스캔. DB 영속 (`buy_signals` 테이블, 전략 간 공유), 최신 데이터 자동 리프레시, 수동 추가. 향후 전략마다 별도 `<N>_<strategy>_Signals.py` 페이지 추가 (scanner만 `SignalScannerPort` 구현체로 추가하면 됨).
  - `main.py` — FastAPI (uvicorn)
  - `minervini_trades_check_cli.py` — 거래 로그 CSV(Ticker, Entry Date…)의 각 진입을 진입 전날 기준 미너비니 STAGE 0~4 로 재판정 + 눈판독용 차트 PNG. 결과 예: `docs/strategy_notes/미너비니_WedgePop거래28건_*.md`
  - `minervini_watchlist_check_cli.py` — 44 스크리너 내보내기 CSV(티커, 산업군, RS…)의 후보 전부를 기준일 종가 기준 STAGE 0~4 재판정 + 차트 PNG + 종목별 도시에(주봉·일봉·실적·헤드라인). 0-3 은 `minervini_checklist_backtest_cli.py live` 결과(`live_json`)를 넘긴다. 결과 예: `docs/strategy_notes/미너비니_2026-10-13_스크리너_*.md`
  - `cache_fill_cli.py` — **캐시 앞단 채우기**. 날짜 단위 Mongo 캐시는 기준일이 거래일 하루만 앞으로 가도 전 종목이 미스가 되어 종목당 Alpha Vantage 1콜(분당 75콜 → 5,000종목 64분)을 다시 했다. `data/adapters/cache_frontier_fill.py` 가 Massive(Polygon) grouped-daily(`BulkDailyBarsPort`, 날짜당 1콜로 전 종목)로 최근 15거래일을 먼저 채우고, 분할 종목은 캐시를 비워 재조회시키며, grouped 에도 없고 최근 어느 소스가 'No data found' 를 기록한 폐지 심볼(SPAC 유닛·워런트 등 약 1,000개)에는 네 캐시 전부에 nodata 마커를 써서 거래일마다 네 소스를 두드리는 비용을 없앤다. 44 스크리너 페이지는 Run 시 자동 호출. 장 마감 후 크론으로 돌려 두면 첫 실행도 빠르다: `python cache_fill_cli.py '{"universe": "nasdaq_full@기준일", "as_of": "YYYY-MM-DD"}'`
  - `evidence_backtest_cli.py` — **근거중심매매(온휘) 규칙 백테스트**: QQQ 로 상승장/하락장/평범 레짐을 월별 분류해 무작위 시작일을 뽑고, 주봉 5대 근거 ≥4 → 일봉 캔들 확인 → 2회 분할 → 근거선 종가 이탈 손절 → 1:1 에서 75% 익절·본전 스탑 → 1.272 청산을 **매일 종가로 재판정**(체결 다음날 시가). 플라세보 2종(같은 날 같은 수 무작위 종목 / 매주 빈 자리 무작위)과 QQQ·SPY 대조. 엔진 `strategy/adapters/evidence_backtest.py` — `SimRules` 기본값은 **2026-10-29 책 재대조판**(1:1 = 3점 추세기반 확장, 2차 = 0.618 이하 레벨 지정가, 지수 주봉 최악 4분면 게이트, 하락 다이버전스 2회 제외, 추격 금지 EMA20+5%, 매도에 MA 없음), 첫 실행판은 `legacy_rules()` / cfg `"preset": "legacy"`, 개별 덮어쓰기 `"rules": {...}`. 결과 `docs/strategy_notes/근거중심매매_백테스트_2026-10.md`, 차이표 체크리스트 문서 §7-2, 원본 `evidence_runs/`
  - `evidence_manual_bt.py` — **근거중심매매 수동(사람 판정) 워크포워드 도구**: `data`(parquet 캐시 + seed 날짜) → `candidates <금요일>`(자동 1차 거름 통과 후보를 기준일까지만 그린 주봉·일봉 차트 `fin_<날짜>.png`, QQQ 포함) → 사람이 M-A~M-H 로 1차 진입 결정 → `holdings <월> <금> '<레벨 json>'`(보유 종목 일봉을 하루씩, 2차 터치·손절·1:1·최종 플래그) → 결정 JSON → `settle`(다음날 시가 체결·수수료·세후·QQQ/SPY). 결과 `docs/strategy_notes/근거중심매매_백테스트_2026-10.md` §3, 결정 파일 `evidence_runs/evidence_manual_*.json`, 차트 `images/evidence_manual_bt_2026_10/`
  - `evidence_recheck_cli.py` — **54 스크리너 내보내기 CSV 재검증 자료**: 기준일 종가(완성 주봉만)로 주봉 예 ≥ N 종목의 자동 판독 + v3 플래그(4분면·하락 다이버 2회·추격 20MA+5%) + 책 레벨(2차 지정가·손절·1:1 3점 확장·최종) + 정크/거래대금 필터 + 4종목씩 차트 → `auto.md`/`rows.json`/`ev_NN.png`. 사람이 §5 수동 체크리스트를 적용해 문서화. 예: `docs/strategy_notes/근거중심매매_2026-10-28_스크리너_226종목_수동검증_v2.md`
  - `growth_dip_scalp_cli.py` — **성장주 당일 눌림 +1% 매매 그리드 배치** (2026-10-30 규칙: 트리거 기본 `open` = 당일 시가 −X%, 청산 기본 `swing` = 당일 +1% 아니면 보유 → +tp2(10%)/다음 실적 1일 전/기한(max_hold_days), 가격 손절 없음; `per_filter=industry_low` = TTM PER < 산업군 중앙값×0.8; 분봉 확정은 `intraday.source=alphavantage`(월별 표본, `data/adapters/alphavantage_intraday_bulk.py` parquet 캐시), `intraday_subgrid` 로 분봉 계산 조합 제한; `build_bars3d` 는 분봉 첫 시가·고가가 일봉과 5% 이상 어긋나는 날을 버림 — 티커 재사용 종목(GEN·WTW)의 AV 분봉이 다른 회사 가격으로 와 가짜 +1,785% 갭이 생겼던 버그; `trades_out` 로 조합별 거래 목록 덤프, `python growth_dip_scalp_cli.py md <json>` 으로 md 만 재생성) (설계서 `docs/strategy_notes/성장주_당일눌림_1퍼센트_설계_2026-10.md`, **결과 `성장주_당일눌림_1퍼센트_검증_2026-10.md` + 쉬운 요약 `…_쉬운요약_2026-10.md` — 기각(R16): 플라세보 A 0/192, 일봉 neutral 은 open 트리거에서 항등식 착시; 소형주(§12, `universe: {"include": ["nasdaq_full"], "exclude": ["sp500", "nasdaq100"]}`) 는 더 나쁨; 손실 방지 필터(§13) `min_days_to_earnings`·`max_vol_ratio`·`trend_filter=sma200+20` 은 **10/31 부터 엔진 기본값 ON** — 이전 실행 재현 시 cfg base 에 0/null/null 명시, `close_pos_stop` 은 기각**): 미너비니 STAGE 2 성장(Alpha Vantage 분기·발표일, point-in-time) 리스트를 매주 고정 → 당일 저가가 전일 종가 −X%(또는 20일선) 트리거 → +tp% 익절 / −sl% 손절 / 당일 종가, 지수 필터·추세 필터·실적일 제외. 일봉은 장중 순서를 모르므로 **세 체결 경계(optimistic/neutral/pessimistic)** 를 같이 내고, `"intraday": {"interval": "5m", "days": 59}` 면 yfinance 5분봉(최근 60일)으로 **확정 체결**을 덧붙인다(Massive 는 플랜상 최근 데이터만). 플라세보 A(리스트 밖 같은 트리거)·B(리스트 종목 무작위 날 시가), QQQ·SPY·리스트 동일비중 보유 대조. cfg `grid` = {필드: [값…]} 데카르트 곱(`max_combos`), 벡터화 `run_sim_fast`(루프판 `run_sim` 과 동일, 테스트로 고정). 엔진 `strategy/adapters/growth_dip_scalp.py`, 테스트 `tests/test_growth_dip_scalp.py`
  - `growth_dip_manual_cli.py` — **자기 PER 변형(설계서 §2.7: `trigger=per_own|per_own_any`, `hold_mode=per`, `per_exit_pct`, `per_lookback`; CLI `pad_days` 420 권장) 의 수동 체크리스트(§3-1) 레짐 표본 점검 도구**: `data`(가격 parquet + 레짐별 표본 달) → `candidates <YYYY-MM> [x] [trigger]`(그 달 적격 트리거를 트리거일까지만 그린 차트·도시에) → 사람 결정 json → `settle`(다음날 시가, PER 회복 종가 청산, 자동 전부/사람 매수/건너뜀 묶음 + QQQ). `features [x] [trigger] [lo] [hi]` 는 전체 적격 이벤트(포지션 제한 없음)를 같은 규칙으로 정산해 진입일 특징 CSV + 손실/이익 3분위·앞뒤 기간 분할 md(`features_md <csv> <x> <trigger>` 로 md 재생성) — 결과 검증 문서 §16(200일선 아래·가격 하락형이 손실 12% vs EPS 상승형 35~42%, 사람 체크리스트와 반대), 설계서 §3-2 데이터판 체크리스트 — 엔진 손잡이 `px_below_med_pct`·`below_sma200_pct`·`per_ref_max_ratio`·`max_last_yoy`·`min_days_since_report`(기본 꺼짐, `checklist_ok` 가 liquid 에 접어 플라세보 A 에도 적용), 그리드 결과 검증 문서 §17(`growth_dip_runs/growth_dip_ck{A,B,B10}_2026-10-31.*`). 결과 검증 문서 §14(엔진 A/B 판)·§15(수동 6 달 46 건 — 체크리스트가 거꾸로 작동), 기록 `growth_dip_runs/growth_dip_manual_2026-10-31/`, 차트 `images/growth_dip_manual_2026_10/`
  - `growth_dip_loss_analysis_cli.py` — 위 그리드의 `trades_out` 거래 목록에 진입일 특징(종가 위치·실적까지 일수·거래량 배수·200일선·QQQ 당일·산업군)을 붙여 이익/손실 거래 비교. 결과 `docs/strategy_notes/growth_dip_runs/growth_dip_loss_analysis_2026-10-31.txt`, 요약 검증 문서 §6-1, 쉬운 요약 `성장주_당일눌림_1퍼센트_쉬운요약_2026-10.md`
  - `optimal_leverage_cli.py` — 나스닥100(^NDX) 연도별 μ·σ·L\*=μ/σ²(성장 최적 노출 배수)와 1배·3배 합성·TQQQ 실제 수익 표. 로드맵 `TQQQ_초과수익_로드맵_2026-10.md` 기둥 3 (Phase 0 결과), 결과 `docs/strategy_notes/tqqq_optimal_leverage_2026-10.md`
  - `vol_target_regime_cli.py` — 로드맵 기둥 2·3 (Phase 1) 배치: 변동성 타기팅(목표σ × 예측기 rv20/rv60/ewma/max) × ⑩ 낙폭 레짐을 세 표본(2010/1999/1986)·10년 롤링·세후로. 엔진 `strategy/adapters/vol_target_regime_lab.py`, 결과 `docs/strategy_notes/변동성타기팅_레짐_검증_2026-10.md`, 원본 `leveraged_etf_runs/vt_regime_*_2026-10-19.*`
  - `crash_early_warning_cli.py` — 로드맵 기둥 2: 거시·신용·VIX·추세 신호의 혼동행렬(⑩ 이탈 대비 선행 일수·거짓 경보)과 "경보 ON 이면 얕은 낙폭에서 이탈" 변형의 세후 성과. 엔진 `strategy/adapters/crash_early_warning.py`, 후속 `reentry_gate_cli.py`(재진입 게이트, 기각). 결과 `docs/strategy_notes/참사_조기경보_검증_2026-10.md`, 가설 장부 `docs/strategy_notes/study_cards/README.md`
  - `scan_cli.py` — 배치 스캔 CLI. JSON config(single or list) → JSON 결과. progress 로그 stderr + `/tmp/batch_partial.json` incremental dump
  - `tests/` — pytest 테스트
- `docker-compose.yaml` — backtester + PostgreSQL(db) 서비스. 컨테이너 기동 시 순서: `alembic upgrade head` → uvicorn + streamlit

## Docker 이미지

`.env` 파일로 Harbor/DockerHub 전환 가능. `.env` 없으면 DockerHub 기본값 사용.
backtester 컨테이너 안에서 uvicorn(8000)과 streamlit(8501)이 함께 실행된다.

## DB 마이그레이션 (Alembic)

스키마 변경 워크플로우:

```bash
# 1) ORM 모델 수정 (signals/adapters/orm.py 등)
# 2) 마이그레이션 자동 생성
docker compose exec backtester alembic revision --autogenerate -m "설명"
# 3) 생성된 alembic/versions/*.py 검토 — autogenerate가 놓치는 케이스 있음
# 4) 적용
docker compose exec backtester alembic upgrade head
# 되돌리기
docker compose exec backtester alembic downgrade -1
```

- 새 ORM 모델 추가 시 반드시 `db.base.Base` 상속 + `alembic/env.py`에서 import (autogenerate 스캔 대상)
- `DATABASE_URL`은 docker-compose가 주입(`postgresql://backtester:backtester@db:5432/backtester`)
- 컨테이너 재시작 시 자동으로 `upgrade head` 실행 — 프로덕션/개발 동일

## 헥사고날 아키텍처 규칙

- 도메인 레이어(`*/domain/`)는 **외부 프레임워크 의존 금지** — SQLAlchemy / yfinance / streamlit 등 import 불가
- 어댑터 레이어(`*/adapters/`)만이 외부 라이브러리에 의존
- 포트(`domain/ports.py`)는 ABC로 선언, 어댑터가 구현
- 새 저장소/데이터소스 추가 시: port 먼저 정의 → 테스트용 in-memory 어댑터 → 프로덕션 어댑터 (DB 등) 순서
- Composition root는 Streamlit page / FastAPI main / CLI — 이 레이어에서만 구현체 선택 후 주입

## Claude Code 설정

- `.claude/settings.json` — 팀 공유 설정 (git 추적)
- `.claude/settings.local.json` — 개인 permissions (git 제외)
