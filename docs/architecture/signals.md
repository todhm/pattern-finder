# Signals 모듈

매수 신호(BuySignal)의 생성·저장·관리를 담당. 헥사고날 구조 예시로 적합 — 도메인은 dataclass + ABC 포트, 어댑터는 SQLAlchemy / Postgres / in-memory / per-strategy scanner.

## 디렉토리

```
signals/
├── domain/
│   ├── models.py             BuySignal, SignalStatus(enum)
│   └── ports.py              SignalRepositoryPort, SignalScannerPort (ABC)
└── adapters/
    ├── in_memory_repo.py     dict 기반, 프로세스 메모리
    ├── postgres_signal_repo.py  SQLAlchemy session 기반 영속 어댑터
    ├── orm.py                BuySignalRow (SQLAlchemy 모델) + 도메인 변환
    ├── universe_scanner.py   Wedge Pop 스캐너 (detector + 엔트리 필터 + 손절/익절 계산)
    ├── bull_flag_scanner.py  Bull Flag 스캐너
    └── wick_play_scanner.py  Wick Play 스캐너 (intraday + 일봉 regime)
```

## 도메인 모델

`BuySignal` — 1개의 매수 기회.

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | str (uuid) | PK |
| `ticker` | str | 종목 심볼 |
| `signal_date` | date | 패턴 발화일 |
| `signal_datetime` | datetime \| None | intraday 신호의 정확한 바 timestamp (tz-aware). 일봉은 `None` (signal_date로 충분) |
| `interval` | str | 바 캐던스 (`"1d"`, `"15m"`, …). 같은 테이블에서 일봉/intraday 워치리스트 분리용 |
| `pattern_name` | str | 예: `wedge_pop`, `bull_flag`, `wick_play` |
| `entry_price` | float | 실제 진입가 (다음 바 open이 있으면 그 값, 없으면 signal close) |
| `stop_loss` | float | 1R 기준 (consolidation low 등 전략별 정의) |
| `metadata` | dict | 근거 데이터 — breakout strength, slope, volume 비율, 필터 게이트, 계산된 stop/TP 레벨, refreshed_at 등 |
| `status` | `SignalStatus` | pending / taken / rejected / expired |
| `notes` | str | 사용자 메모 |
| `created_at` | datetime | 저장 시각 |

> `interval` 기본값은 `"1d"` — 일봉 스캐너는 명시 안 해도 호환된다. ORM 측에는 `server_default="1d"`가 있어 마이그레이션 이전 행도 깨끗하게 업그레이드된다.

## 포트

### `SignalRepositoryPort`
`save / list(status?, interval?) / get / update_status / update_notes / delete` — 6개 메서드.

- `list(interval=None)` → 모든 캐던스 반환 (15m 도입 이전 호환). `interval="1d"` 로 일봉만, `"15m"` 로 intraday만 필터 가능. 페이지마다 자기 캐던스를 명시한다.

### `SignalScannerPort`
`scan(universe, lookback_days, max_tickers=?)` — universe 전체에서 최근 N일 내 signal 추출.

## 어댑터

### `InMemorySignalRepo`
`dict[id -> BuySignal]`. Streamlit 재기동 시 휘발. DB 연결 실패 fallback 용도.

### `PostgresSignalRepo`
SQLAlchemy session 사용. `save`는 `session.merge()`로 UPSERT. 연결 pool은 `db.session.get_engine()` module-level singleton. 인덱스: `signal_date`, `status`, `ticker`, `interval`.

### `UniverseBuySignalScanner` (Wedge Pop)
핵심 파이프라인:

1. `_scan_ticker`: OHLCV fetch → `WedgepopStrategy._with_indicators` → `detector.detect()` → lookback 내 signal 필터 + `volume_ratio ≥ 1.0` 하드 게이트 → `_signal_pressure` (buy/sell ratio)
2. `scan`: 모든 후보를 `(signal_date desc, buy_sell_ratio desc)` 로 정렬 후 signal-바 게이트 (market regime, slope, euphoria cap, close strength, swing breakout) 통과한 것만 `BuySignal`로 변환
3. `refresh_targets(signal)`: 기존 watchlist item의 `metadata`를 최신 바 기준으로 재계산 (HL Trendline 값/slope, resistance supports/hurdles, exhaustion 임계치, latest close). Entry/stop은 불변.
4. `build_signal_at(ticker, date)`: 수동 추가 — 해당 날짜에 detector가 실제 발화한 경우만 `BuySignal` 생성.

### `BullFlagSignalScanner`
Ross Cameron 류 low-float Bull Flag 셋업. fundamentals(float) 게이트 + flag/pole 구조 + 거래량 확인. 사이드바 knob은 `21_Bull_Flag_Strategy` / `22_Multi_Bull_Flag` 의 백테스트 파라미터와 1:1로 대응되어야 한다.

### `WickPlaySignalScanner`
intraday(15m) wick rejection + 일봉 regime 결합. `interval="15m"`로 신호 저장. 워치리스트는 `7_Multi_Wick_Play_Signals` 에서 노출.

## 손절·익절 메타데이터 필드 (Wedge Pop 예시)

실제 `WedgepopStrategy._find_exit` 규칙에 대응:

| metadata key | 대응 exit rule |
|---|---|
| `stop_trendline_at_entry`, `stop_trendline_slope` | `trendline_break` |
| `stop_resistance_supports[].level / .pierce_trigger` | `resistance_break` (downside) |
| `target_resistance_hurdles[].level / .confirm_trigger / .r_multiple` | `resistance_break` (upside) |
| `target_next_resistance`, `r_to_next_resistance` | 위 hurdles 중 최근접 |
| `target_exhaustion_primary`, `r_to_exhaustion_primary` | `exhaustion_exit` primary 경로 임계 |
| `target_exhaustion_rejection`, `r_to_exhaustion_rejection` | rejection override 경로 임계 (× 0.9 leniency) |

각 필드는 strategy flag가 켜져 있을 때만 채워짐 — `enable_trendline_exit` 끄면 HL Trendline 필드 미생성. 다른 전략(Bull Flag, Wick Play)은 자기 exit 규칙에 맞는 별도 metadata 키를 채운다.

## 확장 가이드

**새 저장소 추가** (예: SQLite, S3, Firestore):
1. `signals/adapters/<name>_repo.py` 에 `SignalRepositoryPort` 구현체 작성
2. Streamlit page composition root에서 `st.session_state.signal_repo` 교체
3. UI 코드 수정 없음

**새 전략 스캐너** (예: base-n-break, downside reversal, FVG):
1. `signals/adapters/<name>_scanner.py` 에 `SignalScannerPort` 구현체 작성 — 다른 detector + 다른 필터 조합. intraday면 `interval="15m"` 등 명시
2. 해당 전략용 페이지 생성 (`pages/<N>_<Strategy>_Signals.py`) — 사이드바 knob은 그 전략의 백테스트 페이지와 대응, composition root에서 새 scanner 주입. `list(interval=...)` 로 캐던스 분리
3. watchlist 저장소는 공용이므로 `buy_signals` 테이블에 여러 전략의 신호가 섞여 저장됨. `pattern_name` + `interval` 컬럼으로 구분
4. 백테스트 페이지와 Signals 페이지 사이에 `st.page_link`로 네비게이션 링크 — 같은 전략 묶어서 보기

현재 wire-up 예시:
- `pages/3_Multi_Wedgepop.py` ↔ `pages/4_Multi_Wedgepop_Signals.py` (1d)
- `pages/6_Multi_Wick_Play.py` ↔ `pages/7_Multi_Wick_Play_Signals.py` (15m)
