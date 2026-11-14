"""월간 경기 지표 → 일별 보유 신호. 순수 pandas.

미래 정보 차단: 관측월 m 의 값은 m 월 말 + lag 개월이 지난 다음 달 1 일부터 쓴다(lag=1 → 9월 값은 11월 1일부터).
그 위에 신호는 전날까지 알려진 값으로 만들고(shift 1) 당일 종가에 체결한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RULES = ("yoy_pos", "accel3", "accel6", "yoy_neg", "trough", "accel3_or_pos")


def features(x: pd.Series) -> pd.DataFrame:
    """월간 수준 → 전년 대비(yoy), 그 3·6 개월 변화(accel3·accel6), 지난 5 년 안에서의 yoy 순위(pct, 0~1)."""
    x = x.dropna(); yoy = x / x.shift(12) - 1.0
    return pd.DataFrame({"yoy": yoy, "accel3": yoy - yoy.shift(3), "accel6": yoy - yoy.shift(6), "pct": yoy.rolling(60, min_periods=36).rank(pct=True)})


def to_daily(monthly: pd.DataFrame | pd.Series, days: pd.DatetimeIndex, lag: int = 1) -> pd.DataFrame | pd.Series:
    """관측월 m(월초 날짜 색인) → m + 1 + lag 개월의 1 일부터 유효하게 일별로 펼친다."""
    m = monthly.copy(); m.index = pd.DatetimeIndex(m.index) + pd.DateOffset(months=1 + lag)
    return m.reindex(m.index.union(days)).ffill().reindex(days)


def rule_state(f: pd.DataFrame, rule: str) -> pd.Series:
    """월간 특징 → 보유(1)/현금(0). NaN 인 달은 0."""
    y, a3, a6, p = f["yoy"], f["accel3"], f["accel6"], f["pct"]
    if rule == "yoy_pos":            # 좋아진 걸 확인하고 산다
        s = y > 0
    elif rule == "accel3":           # 나빠도 덜 나빠지면 산다 / 좋아도 덜 좋아지면 판다
        s = a3 > 0
    elif rule == "accel6":
        s = a6 > 0
    elif rule == "yoy_neg":          # 순수 역발상: 나쁠 때만 든다
        s = y < 0
    elif rule == "trough":           # 바닥권(지난 5 년 하위 30%)에서 돌아서면 사서, 꼭지권(상위 30%)에서 꺾이면 판다
        st = np.zeros(len(f)); cur = 0.0
        for i, (pp, aa) in enumerate(zip(p.to_numpy(), a3.to_numpy())):
            if np.isnan(pp) or np.isnan(aa):
                st[i] = cur; continue
            if cur == 0.0 and pp <= 0.3 and aa > 0:
                cur = 1.0
            elif cur == 1.0 and pp >= 0.7 and aa < 0:
                cur = 0.0
            st[i] = cur
        return pd.Series(st, index=f.index)
    elif rule == "accel3_or_pos":    # 좋거나, 좋아지고 있으면 든다
        s = (a3 > 0) | (y > 0)
    else:
        raise ValueError(rule)
    return s.astype(float).where(y.notna(), 0.0)


def flips(w: pd.Series) -> pd.DataFrame:
    """보유 신호의 매수(0→1)·매도(1→0) 날짜."""
    d = w.diff().fillna(0.0)
    return pd.DataFrame({"date": w.index[d != 0], "side": np.where(d[d != 0] > 0, "매수", "매도")})
