"""Performance metrics — pure Python, no extra deps."""
import math
from typing import List, Dict


TRADING_DAYS_PER_YEAR = 252


def calculate_metrics(nav_history: List[Dict], initial_cash: float) -> Dict:
    if not nav_history:
        return _empty_metrics(initial_cash)

    navs = [float(r["nav"]) for r in nav_history]
    if len(navs) < 2:
        return {
            **_empty_metrics(initial_cash),
            "final_nav": navs[0],
            "num_days": 1,
        }

    final = navs[-1]
    total_return = (final - initial_cash) / initial_cash if initial_cash > 0 else 0.0

    days = (nav_history[-1]["date"] - nav_history[0]["date"]).days or 1
    years = days / 365.25
    cagr = (
        (final / initial_cash) ** (1 / years) - 1
        if years > 0 and initial_cash > 0 and final > 0
        else 0.0
    )

    returns = [
        (navs[i] - navs[i - 1]) / navs[i - 1]
        for i in range(1, len(navs))
        if navs[i - 1] > 0
    ]

    if returns:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std_r = math.sqrt(var_r)
        sharpe = (mean_r / std_r) * math.sqrt(TRADING_DAYS_PER_YEAR) if std_r > 0 else 0.0

        downside = [r for r in returns if r < 0]
        if downside:
            dn_std = math.sqrt(sum(r ** 2 for r in downside) / len(downside))
            sortino = (mean_r / dn_std) * math.sqrt(TRADING_DAYS_PER_YEAR) if dn_std > 0 else 0.0
        else:
            sortino = 0.0
    else:
        sharpe = sortino = 0.0

    # MDD
    peak = navs[0]
    mdd = 0.0
    for v in navs:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd

    calmar = cagr / abs(mdd) if mdd < 0 else 0.0

    win_days = sum(1 for r in returns if r > 0)
    win_rate = win_days / len(returns) if returns else 0.0

    return {
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "mdd": round(mdd, 6),
        "calmar": round(calmar, 4),
        "win_rate": round(win_rate, 4),
        "num_days": len(navs),
        "num_returns": len(returns),
        "final_nav": round(final, 2),
        "initial_cash": round(initial_cash, 2),
    }


def _empty_metrics(initial_cash: float) -> Dict:
    return {
        "total_return": 0.0,
        "cagr": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "mdd": 0.0,
        "calmar": 0.0,
        "win_rate": 0.0,
        "num_days": 0,
        "num_returns": 0,
        "final_nav": float(initial_cash),
        "initial_cash": float(initial_cash),
    }
