"""코인 심볼 목록 — Altcoin Momentum(35) / 상세(37) 페이지 공유.

yfinance 심볼 실데이터 검증 완료(2026-08). 이름 충돌 때문에 숫자
접미사가 붙은 심볼이 정본인 경우가 많다 (Hyperliquid=HYPE32196-USD,
Pepe=PEPE24478-USD, TRUMP=TRUMP35336-USD 등).
"""

DEFAULT_COINS = [
    "BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "ADA-USD",
    "SOL-USD", "DOGE-USD", "AVAX-USD", "LINK-USD",
]

# 2023~2025 상장 — {심볼: 상장 연월}
RECENT_COINS = {
    # 2023
    "TAO22974-USD": "2023-03", "ARB11841-USD": "2023-03",
    "PEPE24478-USD": "2023-04", "SUI20947-USD": "2023-05",
    "SEI-USD": "2023-08", "PYTH-USD": "2023-11", "WIF-USD": "2023-12",
    # 2024
    "ONDO-USD": "2024-01", "JUP29210-USD": "2024-01",
    "AERO29270-USD": "2024-02", "VIRTUAL-USD": "2024-02",
    "STRK22691-USD": "2024-02", "W-USD": "2024-03", "ENA-USD": "2024-04",
    "NOT-USD": "2024-05", "HYPE32196-USD": "2024-11", "MOVE32452-USD": "2024-12",
    # 2025
    "S32684-USD": "2025-01", "PENGU34466-USD": "2025-01",
    "TRUMP35336-USD": "2025-01", "BERA-USD": "2025-02", "KAITO-USD": "2025-04",
}
