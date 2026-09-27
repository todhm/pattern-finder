"""수동 체크리스트용 외부 소스 링크 — 티커·기준일로 URL을 조립한다.

2-7(어닝 서프라이즈)·2-8(가이던스)는 "발표일 기준 actual vs estimate"와
"기준일 이전 보도자료"를 직접 봐야 하는 항목이라, 클릭 한 번으로 그 화면에
가도록 링크를 체크리스트 안에 넣는다. 전부 무료/공개 페이지.
"""

from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import quote_plus


def earnings_surprise_links(ticker: str) -> str:
    """2-7 — 분기별 발표일·추정치·실제치·서프라이즈 히스토리."""
    t = ticker.upper()
    return " · ".join([
        f"[Zacks 어닝 히스토리](https://www.zacks.com/stock/research/{t}/earnings-calendar)",
        f"[Nasdaq 서프라이즈](https://www.nasdaq.com/market-activity/stocks/{t.lower()}/earnings)",
        f"[Yahoo 추정치 추이(현재 vs 7·30·60·90일 전)](https://finance.yahoo.com/quote/{t}/analysis)",
        f"[EDGAR 8-K 실적 보도자료](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={t}&type=8-K&owner=include&count=40)",
    ])


def guidance_links(ticker: str, as_of: date | None = None, lookback_days: int = 120) -> str:
    """2-8 — 기준일 **이전** 실적 발표의 가이던스 (look-ahead 방지용 날짜 필터 포함).

    - EDGAR ``dateb``: 기준일 이전 제출분만 나열 → Ex.99.1(보도자료)에 가이던스.
    - 구글 뉴스 기간검색: 기준일 −lookback_days ~ 기준일.
    """
    t = ticker.upper()
    edgar = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={t}"
             f"&type=8-K&owner=include&count=40")
    parts = []
    if as_of is not None:
        edgar += f"&dateb={as_of.strftime('%Y%m%d')}"
        start = as_of - timedelta(days=lookback_days)
        q = quote_plus(f'"{t}" guidance OR outlook')
        gnews = (f"https://www.google.com/search?tbm=nws&q={q}"
                 f"&tbs=cdr:1,cd_min:{start.month}/{start.day}/{start.year},"
                 f"cd_max:{as_of.month}/{as_of.day}/{as_of.year}")
        parts.append(f"[EDGAR 8-K (기준일 {as_of} 이전만)]({edgar})")
        parts.append(f"[구글 뉴스 기간검색 ({start}~{as_of})]({gnews})")
    else:
        parts.append(f"[EDGAR 8-K]({edgar})")
    parts.append(f"[Seeking Alpha 실적 뉴스](https://seekingalpha.com/symbol/{t}/earnings)")
    return " · ".join(parts)
