# Hexagonal Architecture — Ports & Adapters

## 원칙

1. **도메인 레이어**(`*/domain/`)는 외부 프레임워크 의존 금지 — `sqlalchemy`, `yfinance`, `streamlit`, `fastapi` 등 import 불가
2. **포트**(`domain/ports.py`)는 ABC로 선언하는 *인터페이스*. 도메인이 외부 세계에 요구하는 것
3. **어댑터**(`*/adapters/`)는 포트 구현체. 외부 라이브러리 / 인프라 의존은 여기에만
4. **Composition Root** — Streamlit page, FastAPI main, CLI에서만 구현체 선택 후 주입

## 현재 포트 & 어댑터 매핑

| Port | Adapters |
|---|---|
| `data.domain.ports.MarketDataPort` | `YFinanceAdapter`, `CachedMarketDataAdapter` (데코레이터) |
| `data.domain.ports.UniverseProviderPort` | `WikipediaUniverseAdapter` |
| `pattern.domain.ports.PatternDetector` | `WedgePopDetector`, `ExhaustionExtensionTopDetector`, `BaseNBreakDownsideDetector`, ... |
| `strategy.domain.ports.StrategyRunnerPort` | `WedgepopStrategy`, `MultiWedgepopStrategy` |
| `backtest.domain.ports.BacktestEnginePort` | `SimpleBacktestEngine` |
| `visualization.domain.ports.ChartBuilderPort` | `PlotlyChartBuilder` |
| `signals.domain.ports.SignalRepositoryPort` | `PostgresSignalRepo`, `InMemorySignalRepo` |
| `signals.domain.ports.SignalScannerPort` | `UniverseBuySignalScanner` |

## 데코레이터 패턴 예시

`CachedMarketDataAdapter`는 `MarketDataPort`를 구현하면서 내부에 다른 `MarketDataPort`(보통 `YFinanceAdapter`)를 감쌈:

```python
class CachedMarketDataAdapter(MarketDataPort):
    def __init__(self, upstream: MarketDataPort, cache_dir: ...):
        self._upstream = upstream
    
    def fetch_ohlcv(self, symbol, start, end):
        # end >= today → cache 우회 (장중 intraday 갱신)
        # 그 외 → parquet 파일 캐시 사용
```

Composition root:

```python
market_data = CachedMarketDataAdapter(YFinanceAdapter())
```

포트 레벨에서 동일 타입이므로 strategy, scanner는 캐시 유무 모르고 동작.

## 새 어댑터 추가 절차

예: 새 저장소 구현

1. 포트 확인 — 이미 있다면 skip. 없다면 `domain/ports.py`에 ABC 정의
2. 어댑터 파일 생성 — `adapters/<name>.py`
3. 외부 lib import는 이 파일에서만
4. 테스트 어댑터 먼저 (in-memory / fake) 작성하면 domain 테스트 작성 쉬움
5. Composition root (streamlit page 등)에서 선택

## 왜 이렇게 하는가

- **테스트 용이성**: 도메인 로직은 포트에만 의존 → 테스트에서 fake로 교체 쉬움
- **교체 가능성**: yfinance → 다른 data provider, in-memory → Postgres, Plotly → Matplotlib 등 도메인 코드 변경 없이 가능
- **강제된 경계**: 도메인이 SQL / HTTP / UI 세부사항에 오염되지 않음
- **Streamlit 특수사정**: UI 프레임워크가 자주 reload되어도 도메인 로직 영향 없음

## 경고 — 실수하기 쉬운 지점

- `domain/models.py`에서 `sqlalchemy.orm` import하지 말 것. ORM 매핑은 `adapters/orm.py`로 분리 (`BuySignalRow.to_domain()` / `from_domain(...)` 변환기)
- 포트 시그니처에 어댑터-전용 타입 노출 금지 (예: `pd.DataFrame`은 허용되지만 `psycopg2.connection`은 금지)
- Composition root를 domain에 import하지 말 것 (순환 참조 + 테스트 불가)
