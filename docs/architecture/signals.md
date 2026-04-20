# Signals 모듈

매수 신호(BuySignal)의 생성·저장·관리를 담당. 헥사고날 구조 예시로 적합 — 도메인은 dataclass + ABC 포트, 어댑터는 SQLAlchemy / Postgres / in-memory / yfinance scanner.

## 디렉토리

```
signals/
├── domain/
│   ├── models.py      BuySignal, SignalStatus(enum)
│   └── ports.py       SignalRepositoryPort, SignalScannerPort (ABC)
└── adapters/
    ├── in_memory_repo.py        dict 기반, 프로세스 메모리
    ├── postgres_signal_repo.py  SQLAlchemy session 기반 영속 어댑터
    ├── orm.py                   BuySignalRow (SQLAlchemy 모델) + 도메인 변환
    └── universe_scanner.py      스캐너 (detector + 엔트리 필터 + 손절/익절 계산)
```

## 도메인 모델

`BuySignal` — 1개의 매수 기회.

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | str (uuid) | PK |
| `ticker` | str | 종목 심볼 |
| `signal_date` | date | 패턴 발화일 |
| `pattern_name` | str | 예: `wedge_pop` |
| `entry_price` | float | 실제 진입가 (다음 바 open이 있으면 그 값, 없으면 signal close) |
| `stop_loss` | float | 1R 기준 (consolidation low) |
| `metadata` | dict | 근거 데이터 — breakout strength, slope, volume 비율, 필터 게이트, 계산된 stop/TP 레벨, refreshed_at 등 |
| `status` | `SignalStatus` | pending / taken / rejected / expired |
| `notes` | str | 사용자 메모 |
| `created_at` | datetime | 저장 시각 |

## 포트

### `SignalRepositoryPort`
`save / list(status=?) / get / update_status / update_notes / delete` — 6개 메서드. 저장소 구현이 어떻게 되어 있든 UI는 변경 없음.

### `SignalScannerPort`
`scan(universe, lookback_days, max_tickers=?)` — universe 전체에서 최근 N일 내 signal 추출.

## 어댑터

### `InMemorySignalRepo`
`dict[id -> BuySignal]`. Streamlit 재기동 시 휘발. DB 연결 실패 fallback 용도.

### `PostgresSignalRepo`
SQLAlchemy session 사용. `save`는 `session.merge()`로 UPSERT. 연결 pool은 `db.session.get_engine()` module-level singleton.

### `UniverseBuySignalScanner`
핵심 파이프라인:

1. `_scan_ticker`: OHLCV fetch → `WedgepopStrategy._with_indicators` → `detector.detect()` → lookback 내 signal 필터 + `volume_ratio ≥ 1.0` 하드 게이트 → `_signal_pressure` (buy/sell ratio)
2. `scan`: 모든 후보를 `(signal_date desc, buy_sell_ratio desc)` 로 정렬 후 signal-바 게이트 (market regime, slope, euphoria cap, close strength, swing breakout) 통과한 것만 `BuySignal`로 변환
3. `refresh_targets(signal)`: 기존 watchlist item의 `metadata`를 최신 바 기준으로 재계산 (HL Trendline 값/slope, resistance supports/hurdles, exhaustion 임계치, latest close). Entry/stop은 불변.
4. `build_signal_at(ticker, date)`: 수동 추가 — 해당 날짜에 detector가 실제 발화한 경우만 `BuySignal` 생성.

## 손절·익절 메타데이터 필드

실제 `WedgepopStrategy._find_exit` 규칙에 대응:

| metadata key | 대응 exit rule |
|---|---|
| `stop_trendline_at_entry`, `stop_trendline_slope` | `trendline_break` |
| `stop_resistance_supports[].level / .pierce_trigger` | `resistance_break` (downside) |
| `target_resistance_hurdles[].level / .confirm_trigger / .r_multiple` | `resistance_break` (upside) |
| `target_next_resistance`, `r_to_next_resistance` | 위 hurdles 중 최근접 |
| `target_exhaustion_primary`, `r_to_exhaustion_primary` | `exhaustion_exit` primary 경로 임계 |
| `target_exhaustion_rejection`, `r_to_exhaustion_rejection` | rejection override 경로 임계 (× 0.9 leniency) |

각 필드는 strategy flag가 켜져 있을 때만 채워짐 — `enable_trendline_exit` 끄면 HL Trendline 필드 미생성.

## 확장 가이드

**새 저장소 추가** (예: SQLite, S3, Firestore):
1. `signals/adapters/<name>_repo.py` 에 `SignalRepositoryPort` 구현체 작성
2. Streamlit page composition root에서 `st.session_state.signal_repo` 교체
3. UI 코드 수정 없음

**새 패턴 스캐너** (예: 하락 패턴):
1. `SignalScannerPort` 새 구현체 작성 — 다른 detector + 다른 필터 조합
2. 새 페이지 또는 기존 페이지에 추가
