"""방어 바스켓 최적화 실험실 — 그리드 탐색 + 위기 점수.

공격 자산(TQQQ 등)은 고정하고, **방어 사이드를 여러 위기 자산의
바스켓**(주식·금·국채·달러·현금)으로 일반화한다. 바스켓은 일별
고정비중 리밸런싱 인덱스로 합성해 기존
:class:`BandRebalanceStrategy`의 ``defensive_close``로 그대로
주입한다 — 엔진 수정 없이 2-자산 밴드 리밸런싱 + 레짐 필터를
N-자산 방어 바스켓으로 확장하는 구조.

점수 체계 (기준 포트폴리오 = 방어 100% 주식, 동일 설정):

- **성장 점수** = 100 × CAGR / 기준 CAGR — 100이면 기준과 동일한
  복리 성장, 100 초과면 더 빠름.
- **위기 방어 점수**: 위기 구간(닷컴·금융위기·코로나·전쟁 등)
  수익률 r을 기준 수익률 r0와 비교.
  100 = 무손실, 0 = 기준과 같은 손실, 음수 = 기준보다 나쁨,
  100 초과 = 위기에서 오히려 수익 (r ≥ 0일 때 100 + 100r).
- **종합 점수** = (1−λ)×성장 + λ×위기 (λ = 위기 방어 중요도).
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import pandas as pd

# 위기/전쟁 이벤트 구간 — 위기 점수의 채점 대상.
CRISIS_WINDOWS = {
    "닷컴 붕괴 (2000-03~2002-10)": ("2000-03-10", "2002-10-09"),
    "9·11 테러 (2001-09)": ("2001-09-10", "2001-10-11"),
    "이라크전 개전 (2003-03)": ("2003-03-18", "2003-05-01"),
    "금융위기 (2007-10~2009-03)": ("2007-10-09", "2009-03-09"),
    "코로나 쇼크 (2020-02~03)": ("2020-02-19", "2020-03-23"),
    "2022 인플레 약세장 (2021-11~2022-12)": ("2021-11-19", "2022-12-28"),
    "우크라이나 침공 (2022-02~06)": ("2022-02-23", "2022-06-30"),
    "이스라엘-하마스 (2023-10~11)": ("2023-10-06", "2023-11-30"),
}

CASH = "CASH"  # 무수익 현금 leg의 예약 키


def enumerate_weight_grid(
    assets: list[str], step: float = 0.2
) -> list[dict[str, float]]:
    """합이 1인 가중치 조합 전수 나열 (그리드 간격 ``step``).

    예: assets=[금, 국채], step=0.5 → [{금:1}, {금:.5, 국채:.5}, {국채:1}].
    가중치 0인 자산은 결과 dict에서 제외한다.
    """
    units = round(1.0 / step)
    if abs(units * step - 1.0) > 1e-9:
        raise ValueError(f"step {step} must divide 1.0 evenly")
    combos = []
    for combo in combinations_with_replacement(assets, units):
        weights: dict[str, float] = {}
        for a in combo:
            weights[a] = weights.get(a, 0.0) + step
        combos.append(weights)
    return combos


def build_basket_series(
    prices: dict[str, pd.Series | None],
    weights: dict[str, float],
    initial_level: float = 100.0,
) -> pd.Series:
    """일별 고정비중 리밸런싱 바스켓 인덱스를 합성한다.

    ``prices[k] = None``은 현금 leg (일 수익률 0). 가격이 있는
    자산들의 교집합 날짜 위에서, 바스켓 일수익률 = Σ wᵢ·rᵢ
    (매일 목표 비중으로 리밸런싱하는 인덱스 근사 — 방어 사이드는
    ETF 몇 종이라 실무에서도 월 1회 리밸런싱으로 충분히 추종 가능).
    """
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    active = {k: w / total for k, w in weights.items() if w > 0}

    price_legs = {
        k: prices[k] for k in active if prices.get(k) is not None
    }
    if not price_legs:
        # 100% 현금 — 수익률 0의 평평한 인덱스. 달력이 없으므로
        # 아무 가격 시계열이나 요구하는 대신 호출자가 처리하도록 raise.
        raise ValueError("all-cash basket needs at least one priced asset for a calendar")

    joined = pd.DataFrame(price_legs).dropna()
    returns = joined.pct_change().fillna(0.0)
    basket_ret = sum(
        returns[k] * w for k, w in active.items() if k in price_legs
    )
    # 현금 leg(w_cash)는 수익률 0이므로 합산에서 자연히 제외됨.
    return initial_level * (1.0 + basket_ret).cumprod()


def window_return(values: pd.Series, start: str, end: str) -> float | None:
    """구간 수익률. 구간이 데이터 범위 밖이면 None."""
    win = values[start:end]
    if len(win) < 2:
        return None
    return float(win.iloc[-1] / win.iloc[0] - 1.0)


# 위기 구간에서 수익이 났을 때의 보너스 상한. 무제한이면 달러·현금
# 위주의 저성장 조합이 위기 몇 곳의 +수익만으로 성장 열세를 뒤집고
# 1위를 가져가는 왜곡이 생긴다 (MDD-CAGR 지형도에서 두 축 모두
# 밀리는 점에 별이 찍히는 현상).
DEFENSE_BONUS_CAP = 120.0


def defense_score(ret: float, baseline_ret: float) -> float:
    """위기 구간 수익률 → 방어 점수.

    100 = 무손실, 0 = 기준과 동일 손실, 음수 = 기준보다 깊은 손실,
    100~120 = 위기에 수익 (상한 :data:`DEFENSE_BONUS_CAP`). 기준이
    무손실(r0 ≥ 0)이면 절대 손실 기준(100 × (1 + r))으로 대체.
    """
    if ret >= 0:
        return min(100.0 + 100.0 * ret, DEFENSE_BONUS_CAP)
    if baseline_ret >= 0:
        return 100.0 * (1.0 + ret)
    return 100.0 * (1.0 - ret / baseline_ret)


def mdd_score(mdd: float, baseline_mdd: float) -> float:
    """최대 낙폭 → 방어 점수 (위기 '구간'이 못 잡는 전체 낙폭 반영).

    100 = 낙폭 없음, 0 = 기준(주식 방어)과 동일 낙폭, 음수 = 더 깊음.
    입력은 음수 관례 (예: -0.43). 위기 구간 점수와 반씩 섞어 쓰면
    "지정 위기만 피하고 다른 데서 깨지는" 조합의 고득점을 막는다.
    """
    if mdd >= 0:
        return 100.0
    if baseline_mdd >= 0:
        return 100.0 * (1.0 + mdd)
    return 100.0 * (1.0 - mdd / baseline_mdd)


def growth_score(cagr: float, baseline_cagr: float) -> float:
    """CAGR → 성장 점수 (기준 = 100). 기준 CAGR ≤ 0이면 절대
    스케일(100 × (1 + cagr))로 대체."""
    if baseline_cagr <= 0:
        return 100.0 * (1.0 + cagr)
    return 100.0 * cagr / baseline_cagr


def label_weights(weights: dict[str, float], names: dict[str, str]) -> str:
    """{'GC=F': 0.4, 'TLT': 0.6} → '금 40% + 장기국채 60%'."""
    parts = [
        f"{names.get(k, k)} {w:.0%}"
        for k, w in sorted(weights.items(), key=lambda kv: -kv[1])
        if w > 0
    ]
    return " + ".join(parts)
