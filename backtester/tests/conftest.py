import numpy as np
import pandas as pd
import pytest


def _make_df(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[int],
    start_date: str = "2023-01-01",
) -> pd.DataFrame:
    dates = pd.bdate_range(start=start_date, periods=len(closes))
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )


@pytest.fixture
def downtrend_with_reversal() -> pd.DataFrame:
    """Synthetic data: 30-day downtrend followed by a reversal extension bar."""
    np.random.seed(42)
    n = 40

    # Steady downtrend for 30 days: price drifts from 100 to ~70
    closes = [100 - i * 1.0 + np.random.normal(0, 0.3) for i in range(30)]

    # Day 31-35: continue down but more aggressively (extension)
    for i in range(5):
        closes.append(closes[-1] - 2.0 + np.random.normal(0, 0.2))

    # Day 36-40: reversal bars — big bullish candles
    for i in range(5):
        closes.append(closes[-1] + 3.0 + np.random.normal(0, 0.3))

    closes = closes[:n]
    opens = [c + np.random.uniform(-0.5, 0.5) for c in closes]
    # Make reversal bars clearly bullish (close > open)
    for i in range(35, n):
        opens[i] = closes[i] - 2.0

    highs = [max(o, c) + np.random.uniform(0.2, 1.0) for o, c in zip(opens, closes)]
    lows = [min(o, c) - np.random.uniform(0.2, 1.0) for o, c in zip(opens, closes)]

    # Spike volume on reversal bars
    volumes = [1_000_000] * 30 + [800_000] * 5 + [3_000_000] * 5

    return _make_df(opens, highs, lows, closes, volumes[:n])


@pytest.fixture
def consolidation_with_breakout() -> pd.DataFrame:
    """Synthetic data: downtrend, consolidation below MAs, then breakout (wedge pop).

    Deterministic (no random) so the detector conditions are reliably met:
    1. Downtrend wide ranges  → high ATR baseline → ATR_AVG stays elevated.
    2. Consolidation tight ranges → ATR drops → ATR < 0.8 * ATR_AVG at breakout.
    3. Slow drift keeps closes strictly below both EMAs.
    4. First breakout bar just above EMAs with moderate true range.
    """
    # Phase 1 (0-19): steep decline 100 → 70, wide daily ranges
    p1_closes = [100 - i * 1.5 for i in range(20)]

    # Phase 2 (20-44): slow drift 70 → 66.4, tight daily ranges
    p2_closes = [70 - i * 0.15 for i in range(25)]

    # Phase 3 (45-59): breakout from ~69 upward
    p3_closes = [69.0 + i * 1.8 for i in range(15)]

    closes = p1_closes + p2_closes + p3_closes

    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    for i, c in enumerate(closes):
        if i < 20:
            # Downtrend: slightly bearish, wide range
            o = c + 0.3
            opens.append(o)
            highs.append(max(o, c) + 2.5)
            lows.append(min(o, c) - 2.5)
        elif i < 45:
            # Consolidation: slightly bearish, very tight range
            o = c + 0.2
            opens.append(o)
            mid = (o + c) / 2
            highs.append(mid + 0.2)
            lows.append(mid - 0.2)
        else:
            # Breakout: bullish, moderate range
            o = c - 2.5
            opens.append(o)
            highs.append(c + 0.8)
            lows.append(o - 0.5)

    volumes = [1_500_000] * 20 + [600_000] * 25 + [3_000_000] * 15

    return _make_df(opens, highs, lows, closes, volumes)


@pytest.fixture
def flat_market() -> pd.DataFrame:
    """Flat market with no patterns — should produce zero signals."""
    np.random.seed(99)
    n = 60
    closes = [100 + np.random.normal(0, 0.3) for _ in range(n)]
    opens = [c + np.random.normal(0, 0.2) for c in closes]
    highs = [max(o, c) + 0.5 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 0.5 for o, c in zip(opens, closes)]
    volumes = [1_000_000] * n

    return _make_df(opens, highs, lows, closes, volumes)
