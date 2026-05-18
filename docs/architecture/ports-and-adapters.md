# Hexagonal Architecture — Ports & Adapters

## 원칙

1. **도메인 레이어**(`*/domain/`)는 외부 프레임워크 의존 금지 — `sqlalchemy`, `yfinance`, `streamlit`, `fastapi`, `pymongo` 등 import 불가
2. **포트**(`domain/ports.py`)는 ABC로 선언하는 *인터페이스*. 도메인이 외부 세계에 요구하는 것
3. **어댑터**(`*/adapters/`)는 포트 구현체. 외부 라이브러리 / 인프라 의존은 여기에만
4. **Composition Root** — Streamlit page, FastAPI main, CLI에서만 구현체 선택 후 주입. 표준 스택은 `data.adapters.composed_market_data.build_default_market_data()` 팩토리가 캡슐화

## 현재 포트 & 어댑터 매핑

| Port | Adapters |
|---|---|
| `data.domain.ports.MarketDataPort` | `YFinanceAdapter`, `EODHDAdapter`, `MassiveAdapter` (원천) · `CachedMarketDataAdapter`(parquet), `MongoDayCacheAdapter`(Mongo) (캐시 데코레이터) · `FallbackMarketDataAdapter`, `IntervalRoutingMarketData`, `RegularSessionFilter` (라우팅/필터 데코레이터) |
| `data.domain.ports.FundamentalsPort` | `EODHDFundamentalsAdapter`, `MassiveFundamentalsAdapter`, `FallbackFundamentalsAdapter`, `CachedFundamentalsAdapter`, `ComposedFundamentalsAdapter` |
| `data.domain.ports.EarningsCalendarPort` | `EODHDEarningsAdapter` (Matt Diamond Bull Flag catalyst gate) |
| `data.domain.ports.NewsCatalystPort` | `EODHDNewsAdapter` (sentiment-aware news catalyst gate) |
| `data.domain.ports.UniverseProviderPort` | `WikipediaUniverseAdapter` (S&P 500 / Nasdaq-100 / KOSPI/KOSDAQ 등) |
| `pattern.domain.ports.PatternDetector` | `WedgePopDetector`, `WedgeDropDetector`, `ExhaustionExtensionTopDetector`, `BaseNBreakDownsideDetector`, `EmaCrossbackDownsideDetector`, `ReversalExtensionDetector`, `BullFlagDetector`, `MattDiamondBullFlagDetector`, `WickPlayDetector`, `FairValueGapDetector`, `FirstCandleRuleDetector`, `ScrafaceOrbDetector`, `TradeSharpOrbDetector` |
| `strategy.domain.ports.StrategyRunnerPort` | `WedgepopStrategy`, `Wedgepop15mStrategy`, `MultiWedgepopStrategy`, `WickPlayStrategy`, `WickPlay15mStrategy`, `MultiWickPlayStrategy`, `BullFlagStrategy`, `MultiBullFlagStrategy`, `MattDiamondBullFlagStrategy`, `FairValueGapStrategy`, `MultiFairValueGapStrategy`, `FirstCandleRuleStrategy`, `MultiFirstCandleStrategy`, `ScrafaceStrategy`, `TradeSharpStrategy` |
| `backtest.domain.ports.BacktestEnginePort` | `SimpleBacktestEngine` |
| `visualization.domain.ports.ChartBuilderPort` | `PlotlyChartBuilder` |
| `signals.domain.ports.SignalRepositoryPort` | `PostgresSignalRepo`, `InMemorySignalRepo` |
| `signals.domain.ports.SignalScannerPort` | `UniverseBuySignalScanner` (Wedgepop), `BullFlagSignalScanner`, `WickPlaySignalScanner` |

> 시장 캘린더/세션 정의는 `data.domain.market_calendar`(NY/KR 등 `MarketCalendar` 값 객체 + `market_for_ticker`)에 모여 있다. 도메인 모듈이지만 외부 의존이 없어 detector/strategy/chart 모두에서 임포트한다.

## 데코레이터 패턴 — 시장 데이터 스택

`MarketDataPort`는 동일 인터페이스 데코레이터를 여러 단계로 합성한다. `build_default_market_data()`가 만드는 표준 스택:

```
IntervalRoutingMarketData(
    sub_daily = FallbackMarketDataAdapter(
        primary  = MongoDayCacheAdapter(EODHDAdapter,    "bars_eodhd"),
        fallback = MongoDayCacheAdapter(MassiveAdapter,  "bars_massive"),
    ),
    daily     = FallbackMarketDataAdapter(
        primary  = MongoDayCacheAdapter(YFinanceAdapter, "bars_yfinance"),
        fallback = MongoDayCacheAdapter(MassiveAdapter,  "bars_massive"),
    ),
)
```

설계 의도:

- **IntervalRoutingMarketData** — `1d/1wk/1mo`는 무료/넓은 depth의 yfinance로, intraday(`15m/5m/1h` 등)는 EODHD/Massive로 라우팅
- **FallbackMarketDataAdapter** — primary가 quota/HTTP 에러를 던지면 fallback으로 투명 전환
- **MongoDayCacheAdapter** — 일자 단위(per-symbol/interval/date) Mongo 캐시. 컨테이너 재빌드/머신 간 공유. `bypass_today=True`면 당일은 항상 refetch
- **CachedMarketDataAdapter** — parquet 디스크 캐시(legacy). signal 페이지(예: `4_Multi_Wedgepop_Signals`)는 intraday 갱신 정책이 다르므로 여전히 이 어댑터를 직접 씀
- **RegularSessionFilter** — RTH(09:30–16:00 ET 등) 외 바를 제거. 어댑터 시그니처가 동일해 어디서나 끼워 넣을 수 있음

포트 레벨에서 모두 같은 타입이므로 strategy/scanner는 캐시/라우팅/필터 유무를 알 필요가 없다.

### Fundamentals 스택

`FundamentalsPort`도 동일 패턴: `ComposedFundamentalsAdapter`가 `FallbackFundamentalsAdapter(primary=EODHD, fallback=Massive)`를 `CachedFundamentalsAdapter`(다일 TTL)로 감싼다. Bull Flag 전략의 float-share 필터 등에서 사용.

## 새 어댑터 추가 절차

예: 새 저장소 / 데이터 소스 구현

1. 포트 확인 — 이미 있다면 skip. 없다면 `domain/ports.py`에 ABC 정의
2. 어댑터 파일 생성 — `adapters/<name>.py`
3. 외부 lib import는 이 파일에서만
4. 테스트 어댑터 먼저 (in-memory / fake) 작성하면 domain 테스트 작성 쉬움
5. Composition root (Streamlit page, `composed_market_data` 팩토리, CLI)에서 선택/주입

## 왜 이렇게 하는가

- **테스트 용이성**: 도메인 로직은 포트에만 의존 → 테스트에서 fake로 교체 쉬움
- **교체 가능성**: yfinance → EODHD → Massive, in-memory → Postgres, parquet → Mongo, Plotly → Matplotlib 등 도메인 코드 변경 없이 가능
- **강제된 경계**: 도메인이 SQL / HTTP / UI 세부사항에 오염되지 않음
- **장애 격리**: 데코레이터가 동일 포트 타입이므로 quota 초과·rate limit이 와도 라우팅/폴백을 한 자리에서 처리

## 경고 — 실수하기 쉬운 지점

- `domain/models.py`에서 `sqlalchemy.orm` import하지 말 것. ORM 매핑은 `adapters/orm.py`로 분리 (`BuySignalRow.to_domain()` / `from_domain(...)` 변환기)
- 포트 시그니처에 어댑터-전용 타입 노출 금지 (예: `pd.DataFrame`은 허용되지만 `pymongo.Collection` / `psycopg2.connection`은 금지)
- Composition root를 domain에 import하지 말 것 (순환 참조 + 테스트 불가)
- 페이지에서 어댑터를 직접 hand-wire 대신 `build_default_market_data()` 팩토리를 호출 — 라우팅/캐시 정책이 페이지마다 어긋나는 것을 막는다 (signal 페이지의 `bypass_today=True` 우회 예외만 인지)
