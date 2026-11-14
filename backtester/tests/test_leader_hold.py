import numpy as np

from strategy.adapters.leader_hold import HoldRules, run_leader_hold


def _mk(T=30):
    a = 100 * 1.02 ** np.arange(T)                               # 계속 오름
    b = np.where(np.arange(T) < 10, 100 * 1.03 ** np.arange(T), 100 * 1.03 ** 10 * 0.9 ** (np.arange(T) - 10))   # 10일 뒤 급락
    c = np.column_stack([a, b]); o = np.vstack([c[:1], c[:-1]])  # 시가 = 전날 종가
    return c, o, np.ones_like(c, bool)


def test_buy_next_open_and_hold_winner_no_cost():
    c, o, alive = _mk(); elig = np.zeros_like(alive); elig[0, 0] = True
    eq, tr = run_leader_hold(c, o, alive, elig, c, HoldRules(slots=1, cost=0.0), 0, 29)
    assert eq[0] == 1.0                                          # 신호일에는 아직 현금
    assert np.isclose(eq[-1], c[-1, 0] / o[1, 0])                # 다음날 시가 매수 → 끝까지
    assert tr == [(0, 1, -1, c[-1, 0] / o[1, 0] - 1.0)]


def test_trail_exit_sells_next_open_and_refills():
    c, o, alive = _mk(); elig = np.zeros_like(alive); elig[0, 1] = True; elig[15:, 0] = True
    eq, tr = run_leader_hold(c, o, alive, elig, c, HoldRules(slots=1, trail=0.15, cost=0.0), 0, 29)
    j, ent, ex, r = tr[0]
    assert (j, ent) == (1, 1) and ex == 13                       # 종가가 고점 -15% 아래로 간 날(12) 다음 시가
    assert np.isclose(r, o[13, 1] / o[1, 1] - 1) and tr[1][0] == 0 and tr[1][2] == -1


def test_exit_line_delisting_and_cost():
    c, o, alive = _mk(); alive[20:, 0] = False; c[20:, 0] = c[19, 0]; o[20:, 0] = c[19, 0]
    elig = np.zeros_like(alive); elig[0, 0] = True
    eq, tr = run_leader_hold(c, o, alive, elig, c, HoldRules(slots=2, cost=0.01), 0, 29)
    assert tr[0][2] == 21                                        # 폐지 확인(20) 다음날 마지막 종가에 현금화
    assert np.isclose(eq[-1], 0.5 + 0.5 * 0.99 * (c[19, 0] / o[1, 0]) * 0.99)   # 두 칸 중 한 칸만 사용, 양쪽 비용
    line = np.full_like(c, 1e9)                                  # 이탈선이 항상 위 → 산 날 바로 매도 신호
    _, tr2 = run_leader_hold(c, o, np.ones_like(alive), elig, c, HoldRules(slots=1, cost=0.0), 0, 29, exit_line=line)
    assert tr2[0][1:3] == (1, 2)


def test_random_pick_is_reproducible():
    c, o, alive = _mk(); elig = np.ones_like(alive)
    a = run_leader_hold(c, o, alive, elig, c, HoldRules(slots=1), 0, 29, rng=np.random.default_rng(3))[0]
    b = run_leader_hold(c, o, alive, elig, c, HoldRules(slots=1), 0, 29, rng=np.random.default_rng(3))[0]
    assert np.allclose(a, b)
