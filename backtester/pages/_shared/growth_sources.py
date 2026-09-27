"""성장 펀더멘털 소스 체인 — 44(스크리너)·45(리플레이) 공용 composition helper.

Alpha Vantage(기본 — 분기·연간 + 발표일) → EODHD(플랜에 fundamentals
있을 때만). 앞 소스가 빈 결과면 다음 소스를 시도한다. 키가 없어
어댑터 생성이 실패하면 그 소스는 조용히 건너뛴다.
"""

from __future__ import annotations

from data.domain.models import GrowthSnapshot
from data.domain.ports import GrowthFundamentalsPort


def build_growth_adapters() -> list[GrowthFundamentalsPort]:
    adapters: list[GrowthFundamentalsPort] = []
    try:
        from data.adapters.alphavantage_growth_fundamentals import (
            AlphaVantageGrowthFundamentalsAdapter,
        )
        adapters.append(AlphaVantageGrowthFundamentalsAdapter())
    except Exception:
        pass
    try:
        from data.adapters.eodhd_growth_fundamentals import (
            EODHDGrowthFundamentalsAdapter,
        )
        adapters.append(EODHDGrowthFundamentalsAdapter())
    except Exception:
        pass
    return adapters


def fetch_growth_snapshot(
    ticker: str, adapters: list[GrowthFundamentalsPort] | None = None,
) -> GrowthSnapshot | None:
    """소스 체인 순서대로 시도, 분기 데이터가 있는 첫 스냅샷 반환.

    전부 비면 마지막 스냅샷(빈 리스트)을, 어댑터가 하나도 없으면 None.
    """
    adapters = build_growth_adapters() if adapters is None else adapters
    snap: GrowthSnapshot | None = None
    for adapter in adapters:
        snap = adapter.fetch(ticker)
        if snap.quarters:
            break
    return snap
