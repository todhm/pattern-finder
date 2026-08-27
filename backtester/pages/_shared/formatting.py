"""페이지 공용 표시 포맷 헬퍼."""

from __future__ import annotations


def fmt_price(p: float) -> str:
    """가격 표시 — 저가(분할 조정) 종목에서 자릿수 뭉개짐 방지.

    TQQQ 2010년 조정가 ~$0.4, SOXL ~$0.3처럼 $10 미만 가격은 소수
    둘째 자리로는 +1% 익절 목표가가 체결가와 똑같이 보인다
    (0.6659 vs 0.6739 → 둘 다 "0.67"). $10 미만은 4자리, 이상은
    2자리로 표시한다.
    """
    if not p:
        return ""
    if abs(p) < 10:
        return f"{p:,.4f}"
    return f"{p:,.2f}"
