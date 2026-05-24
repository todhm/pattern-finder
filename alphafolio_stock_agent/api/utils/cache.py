"""
Cache utility functions for stock-detail page caching
"""
from datetime import datetime, timedelta


def calculate_ttl(market: str) -> int:
    """
    Calculate TTL (time-to-live) in seconds until cache expiration.

    Korean market (KR):
    - Mon-Thu after 21:00: expires next day 20:59
    - Fri after 21:00: expires Monday 20:59

    US market (US):
    - Tue-Fri after 13:00: expires next day 12:59
    - Sat after 13:00: expires Tuesday 12:59

    Args:
        market: "KR" or "US"

    Returns:
        TTL in seconds
    """
    now = datetime.now()
    weekday = now.weekday()  # 0=Mon, 1=Tue, ..., 6=Sun

    if market == "KR":
        # Korean market: expires at 20:59
        expire_hour = 20
        expire_minute = 59

        if weekday == 4:  # Friday
            # Expires Monday 20:59 (3 days later)
            days_to_add = 3
        elif weekday == 5:  # Saturday
            # Expires Monday 20:59 (2 days later)
            days_to_add = 2
        elif weekday == 6:  # Sunday
            # Expires Monday 20:59 (1 day later)
            days_to_add = 1
        else:
            # Mon-Thu: expires next day 20:59
            days_to_add = 1

        expire_at = (now + timedelta(days=days_to_add)).replace(
            hour=expire_hour,
            minute=expire_minute,
            second=0,
            microsecond=0
        )
    else:
        # US market: expires at 12:59
        expire_hour = 12
        expire_minute = 59

        if weekday == 5:  # Saturday
            # Expires Tuesday 12:59 (3 days later)
            days_to_add = 3
        elif weekday == 6:  # Sunday
            # Expires Tuesday 12:59 (2 days later)
            days_to_add = 2
        elif weekday == 0:  # Monday
            # Expires Tuesday 12:59 (1 day later)
            days_to_add = 1
        else:
            # Tue-Fri: expires next day 12:59
            days_to_add = 1

        expire_at = (now + timedelta(days=days_to_add)).replace(
            hour=expire_hour,
            minute=expire_minute,
            second=0,
            microsecond=0
        )

    ttl_seconds = int((expire_at - now).total_seconds())

    # Ensure minimum TTL of 60 seconds
    return max(ttl_seconds, 60)


def get_cache_key_detail(symbol: str) -> str:
    """Get cache key for stock detail page data"""
    return f"stock:detail:{symbol}"


def get_cache_key_chart(symbol: str, time_range: str) -> str:
    """Get cache key for chart data"""
    return f"stock:chart:{symbol}:{time_range}"


def get_cache_key_strategy(symbol: str) -> str:
    """Get cache key for strategy data"""
    return f"stock:strategy:{symbol}"
