"""AI 패턴 백테스트 — 설정한 기간을 봉 순서대로 걸어가며 모델에게 진입·청산을 묻고 손익을 낸다 (LLM 을 모르는 순수 pandas).

걷는 방식 (워크포워드, 뒤 봉은 모델에게 절대 안 간다):

1. 기간 안에서 ``step`` 봉마다 '판단 시점'을 잡는다(분봉이고 ``regular_only`` 면 정규장 봉만 센다).
2. 포지션이 없으면 그 시점까지의 자료(``build_packet``)로 진입 판단을 받는다. 답이 enter_long 이면
   market = 다음 봉 시가 / breakout_close = 다음 판단 시점 전까지 trigger 위 마감 봉의 종가 / limit = trigger 까지 내려오면 그 가격에 체결.
   다음 판단 시점이 오면 앞의 조건부 주문은 버리고 새로 묻는다.
3. 포지션이 있으면 봉마다 손절·목표를 기계적으로 적용하고, ``manage`` 가 켜져 있으면 ``step`` 봉마다 모델에게 '계속 보유 / 다음 봉 시가 청산 /
   손절 올리기 / 목표 바꾸기'를 묻는다. ``flat_at_close`` 면 그날 마지막 봉 종가에 청산, ``max_hold`` 봉이 지나면 청산.
4. 청산된 봉이 마감된 뒤의 첫 판단 시점부터 다시 진입을 묻는다. 한 번에 한 포지션, 전액 투입, 레버리지 없음.

API 호출은 호출 쪽이 넘긴 ``ask_entry(packets) -> [(Judgement, usage)]`` (여러 장을 한꺼번에 — 병렬·캐시는 호출 쪽 일)와
``ask_manage(packet) -> (Manage, usage)`` 가 한다. 진입 판단은 ``chunk`` 개씩 미리 받아 두고, 그 안에서 체결이 나면 보유 구간에 든 나머지는 버린다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd

from strategy.adapters.ai_chart_judge import NY, ChartSpec, Judgement, Manage, Packet, ScanResult, build_packet


@dataclass
class BtConfig:
    step: int = 4                    # 몇 봉마다 묻나
    max_hold: int = 0                # 최대 보유 봉 (0 = 없음)
    fee_bps: float = 5.0             # 편도
    blind: bool = True
    manage: bool = True              # 보유 중에도 step 봉마다 묻기
    flat_at_close: bool = False      # 그날 마지막 봉 종가에 청산 (분봉)
    regular_only: bool = False       # 정규장 봉에서만 묻기 (분봉, 미국 주식)
    ask_from: str = ""               # 진입을 묻는 시간대 (뉴욕 "HH:MM", 분봉만; 빈 값 = 제한 없음). 보유 중 관리 판단은 시간대와 무관
    ask_until: str = ""
    min_history: int = 40            # 판단에 필요한 최소 앞 봉 수
    chunk: int = 6                   # 진입 판단을 한 번에 몇 장씩 받아 둘지
    max_calls: int = 200             # 호출 상한 (넘으면 거기서 멈춘다)


# ---------------------------------------------------------------------------
# 체결·보유
# ---------------------------------------------------------------------------
def find_fill(j: Judgement, F: pd.DataFrame, i0: int, valid: int) -> dict:
    """판정 ``j`` 가 ``F`` 의 위치 i0 (판단 봉 바로 다음 봉)부터 ``valid`` 봉 안에 체결되는지. F 는 Open/High/Low/Close."""
    if j.decision != "enter_long" or j.entry_type == "none":
        return {"filled": False, "why": "진입 안 함"}
    if j.stop is None:
        return {"filled": False, "why": "손절가 없음 (무효)"}
    if i0 >= len(F):
        return {"filled": False, "why": "뒤 봉 없음"}
    if j.entry_type == "market":
        i, entry, start = i0, float(F["Open"].iloc[i0]), i0        # 시가에 샀으니 진입 봉 자체도 손절·목표를 본다
    else:
        if j.entry_trigger is None:
            return {"filled": False, "why": "진입 가격 없음 (무효)"}
        seg = F.iloc[i0: i0 + max(1, valid)]
        if j.entry_type == "breakout_close":
            hit = np.flatnonzero(seg["Close"].to_numpy() > j.entry_trigger)
            if not len(hit):
                return {"filled": False, "why": f"{len(seg)}봉 안에 돌파 마감 없음"}
            i = i0 + int(hit[0]); entry, start = float(F["Close"].iloc[i]), i + 1
        else:                                                      # limit: 눌림 매수
            hit = np.flatnonzero(seg["Low"].to_numpy() <= j.entry_trigger)
            if not len(hit):
                return {"filled": False, "why": f"{len(seg)}봉 안에 매수 가격까지 안 내려옴"}
            i = i0 + int(hit[0]); entry, start = min(float(F["Open"].iloc[i]), float(j.entry_trigger)), i
    if j.stop >= entry:
        return {"filled": False, "why": "손절가가 진입가 이상 (무효)"}
    return {"filled": True, "i": i, "entry": entry, "start": start, "limit_bar": j.entry_type == "limit"}


def walk_position(F: pd.DataFrame, fill: dict, j: Judgement, *, fee_bps: float = 5.0, max_hold: int = 0, flat_last: Optional[np.ndarray] = None,
                  manage_step: int = 0, manage_cb: Optional[Callable[[int, dict], Optional[Manage]]] = None) -> dict:
    """체결된 포지션을 봉마다 걸어 청산까지. ``flat_last[t]`` 가 참인 봉은 그날 마지막 봉(종가 청산). ``manage_cb(t, pos)`` 는 t 봉 마감 뒤 묻는다."""
    i0, entry = fill["i"], fill["entry"]
    stop, t1, t2 = float(j.stop), j.target1, j.target2
    if t1 is not None and t1 <= entry:                      # 체결가가 이미 목표 위(갭 상승 등)면 그 목표는 버린다 — 진입 즉시 청산되는 것을 막는다
        t1 = None
    if t2 is not None and t2 <= max(entry, t1 or entry):
        t2 = None
    risk = entry - stop
    legs: list[tuple[float, float, str, int]] = []          # (비중, 가격, 사유, 봉 위치)
    left = 1.0; t1_done = t1 is None
    mfe = mae = 0.0
    notes: list[dict] = []
    n = len(F); t = fill["start"]; last_t = i0
    while t < n and left > 0:
        o, h, l, c = (float(F[k].iloc[t]) for k in ("Open", "High", "Low", "Close"))
        last_t = t
        same_bar_limit = fill.get("limit_bar") and t == i0   # 눌림 체결 봉: 체결 뒤의 고가는 알 수 없으니 목표는 안 보고 손절만 본다
        mfe = max(mfe, (c if same_bar_limit else h) / entry - 1); mae = min(mae, l / entry - 1)
        if j.stop_basis == "touch" and l <= stop:
            legs.append((left, min(o, stop) if not same_bar_limit else stop, "손절", t)); left = 0.0; break
        if not same_bar_limit:
            if not t1_done and h >= t1:
                part = left if t2 is None else left / 2
                legs.append((part, max(o, t1), "목표1", t)); left -= part; t1_done = True
            if left > 0 and t1_done and t2 is not None and h >= t2:
                legs.append((left, max(o, t2), "목표2", t)); left = 0.0; break
            if left <= 0:
                break
        if j.stop_basis == "close" and c < stop:
            legs.append((left, c, "손절", t)); left = 0.0; break
        if flat_last is not None and flat_last[t]:
            legs.append((left, c, "장 마감", t)); left = 0.0; break
        if max_hold and t - i0 >= max_hold:
            legs.append((left, c, "시간 청산", t)); left = 0.0; break
        if manage_cb is not None and manage_step > 0 and t > i0 and (t - i0) % manage_step == 0:
            m = manage_cb(t, {"entry": entry, "stop": stop, "target": (t1 if not t1_done else t2), "bars_held": t - i0})
            if m is not None:
                notes.append({"t": t, "action": m.action, "new_stop": m.new_stop, "new_target": m.new_target, "confidence": m.confidence, "reason": m.reason})
                if m.action == "exit":
                    px, tt = (float(F["Open"].iloc[t + 1]), t + 1) if t + 1 < n else (c, t)
                    legs.append((left, px, "AI 청산", tt)); left = 0.0; last_t = tt; break
                if m.new_stop is not None and stop < m.new_stop < c:
                    stop = float(m.new_stop)                 # 손절은 올리기만 한다
                if m.new_target is not None and m.new_target > c:
                    if not t1_done:
                        t1 = float(m.new_target)
                    else:
                        t2 = float(m.new_target)
        t += 1
    if left > 0:
        legs.append((left, float(F["Close"].iloc[last_t]), "기간 끝" if last_t == n - 1 else "시간 청산", last_t)); left = 0.0
    fee = 2 * fee_bps / 1e4
    gross = sum(w * (px / entry - 1) for w, px, _, _ in legs)
    return {"filled": True, "entry_i": i0, "entry_time": F.index[i0], "entry": entry, "exit_time": F.index[legs[-1][3]], "exit_i": legs[-1][3],
            "legs": [{"비중": w, "가격": px, "사유": why, "시각": F.index[i], "i": i} for w, px, why, i in legs],
            "why": " → ".join(why for _, _, why, _ in legs), "ret": gross - fee, "R": (gross - fee) * entry / risk, "risk_pct": risk / entry,
            "stop0": float(j.stop), "stop_last": stop, "mfe": mfe, "mae": mae, "bars": legs[-1][3] - i0, "manage_notes": notes}


# ---------------------------------------------------------------------------
# 기간 백테스트
# ---------------------------------------------------------------------------
def _ny_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (idx.tz_localize(NY) if idx.tz is None else idx.tz_convert(NY)).tz_localize(None)


def checkpoints(bars: pd.DataFrame, spec: ChartSpec, cfg: BtConfig, start, end) -> list[int]:
    """판단 시점(봉 위치) 목록. 마지막 봉은 뒤에 체결할 봉이 없으므로 뺀다."""
    idx = bars.index
    lo = pd.Timestamp(start); hi = pd.Timestamp(end)
    if idx.tz is not None:
        lo = lo.tz_localize(idx.tz) if lo.tzinfo is None else lo.tz_convert(idx.tz)
        hi = hi.tz_localize(idx.tz) if hi.tzinfo is None else hi.tz_convert(idx.tz)
    ok = (idx >= lo) & (idx <= hi) & (np.arange(len(idx)) >= cfg.min_history) & (np.arange(len(idx)) < len(idx) - 1)
    if spec.intraday and spec.session == "us" and cfg.regular_only:
        x = _ny_naive(idx); mins = x.hour * 60 + x.minute
        ok &= (mins >= 9 * 60 + 30) & (mins < 16 * 60)
    if spec.intraday and (cfg.ask_from or cfg.ask_until):
        x = _ny_naive(idx); mins = x.hour * 60 + x.minute
        hm = lambda s: int(s[:2]) * 60 + int(s[3:5])  # noqa: E731
        if cfg.ask_from:
            ok &= mins >= hm(cfg.ask_from)
        if cfg.ask_until:
            ok &= mins <= hm(cfg.ask_until)
    elig = np.flatnonzero(ok)
    return [int(i) for i in elig[:: max(1, cfg.step)]]


def _flat_last(bars: pd.DataFrame, spec: ChartSpec, cfg: BtConfig) -> Optional[np.ndarray]:
    if not (spec.intraday and cfg.flat_at_close):
        return None
    x = _ny_naive(bars.index); day = x.normalize().to_numpy()
    last = np.r_[day[1:] != day[:-1], True]                      # 그날의 마지막 봉
    if spec.session == "us" and cfg.regular_only:                # 정규장만 보면 정규장 마지막 봉(15:45 등)이 마감 봉
        mins = np.asarray(x.hour * 60 + x.minute)
        reg = (mins >= 9 * 60 + 30) & (mins < 16 * 60)
        nxt_reg_same_day = np.r_[(reg[1:] & (day[1:] == day[:-1])), False]
        last = last | (reg & ~nxt_reg_same_day)
    return last


def run_backtest(bars: pd.DataFrame, spec: ChartSpec, cfg: BtConfig, *, start, end,
                 ask_entry: Callable[[list[Packet]], list[tuple[Judgement, dict]]],
                 ask_manage: Optional[Callable[[Packet], tuple[Manage, dict]]] = None,
                 context: pd.DataFrame | None = None, ticker: str = "", progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """기간 [start, end] 를 걸어 거래·판정·자본 곡선·성적을 돌려준다. bars 는 기간 앞의 여유 봉(차트용)을 포함해도 된다."""
    f = bars[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()
    F = f[["Open", "High", "Low", "Close"]]
    cps = checkpoints(f, spec, cfg, start, end)
    flat_last = _flat_last(f, spec, cfg)
    decisions: list[dict] = []; trades: list[dict] = []
    calls = 0; cost = 0.0; stopped = ""

    def packet(i: int, position: dict | None = None) -> Packet:
        return build_packet(f.iloc[: i + 1], f.index[i], spec=spec, context=context, ticker=ticker, blind=cfg.blind, position=position)

    def manage_cb(t: int, pos: dict) -> Optional[Manage]:
        nonlocal calls, cost
        if ask_manage is None or calls >= cfg.max_calls:
            return None
        pk = packet(t, pos)
        m, u = ask_manage(pk)
        calls += 0 if u.get("cached") else 1; cost += u.get("cost_usd", 0.0)
        return m.to_real(pk.scale, text_ref=pk.last)

    p = 0
    while p < len(cps):
        if calls >= cfg.max_calls:
            stopped = f"호출 상한 {cfg.max_calls}회에 닿아 {f.index[cps[p]]} 에서 멈춤"
            break
        batch = cps[p: p + max(1, cfg.chunk)]
        if progress:
            progress(p / len(cps), f"{_ny_naive(f.index[[batch[0]]])[0]:%Y-%m-%d %H:%M} 부터 {len(batch)}개 시점 판단 중 · 거래 {len(trades)}건 · 호출 {calls}")
        pks = [packet(i) for i in batch]
        answers = ask_entry(pks)
        for _, u in answers:                                     # 체결 뒤 버려지는 답도 낸 호출이므로 센다
            calls += 0 if u.get("cached") else 1; cost += u.get("cost_usd", 0.0)
        jumped = False
        for k, (i, pk, (j_raw, u)) in enumerate(zip(batch, pks, answers)):
            j = j_raw.to_real(pk.scale, text_ref=pk.last)
            nxt = cps[p + k + 1] if p + k + 1 < len(cps) else len(f) - 1
            valid = max(1, nxt - i)                              # 조건부 주문은 다음 판단 시점까지만 산다
            fill = find_fill(j, F, i + 1, min(valid, j.trigger_valid_bars) if j.trigger_valid_bars > 0 else valid)
            d = {"i": i, "time": f.index[i], "close": float(f["Close"].iloc[i]), "judgement": j, "usage": u, "filled": fill["filled"], "why": fill.get("why", "")}
            decisions.append(d)
            if not fill["filled"]:
                continue
            r = walk_position(F, fill, j, fee_bps=cfg.fee_bps, max_hold=cfg.max_hold, flat_last=flat_last,
                              manage_step=cfg.step if cfg.manage else 0, manage_cb=manage_cb if cfg.manage else None)
            r.update({"decision_i": i, "decision_time": f.index[i], "setup": j.setup, "confidence": j.confidence, "judgement": j})
            trades.append(r); d["why"] = r["why"]; d["ret"] = r["ret"]
            q = p + k + 1
            while q < len(cps) and cps[q] < r["exit_i"]:         # 청산 봉이 마감된 뒤의 시점부터 다시 묻는다
                q += 1
            p = q; jumped = True
            break
        if not jumped:
            p += len(batch)
    if progress:
        progress(1.0, "정리 중")

    return _finish(f, trades, decisions, cps, cfg.step, calls, cost, stopped)


def _finish(f: pd.DataFrame, trades: list[dict], decisions: list[dict], cps: list[int], step: int, calls: int, cost: float, stopped: str) -> dict:
    """거래 목록 → 자본 곡선·성적."""
    cur = 1.0
    # 자본 곡선 (전액 투입, 봉 종가 평가)
    eq = np.ones(len(f)); prev_end = 0
    close = f["Close"].to_numpy()
    for tr in trades:
        a, b, entry = tr["entry_i"], tr["exit_i"], tr["entry"]
        eq[prev_end: a] = cur
        legs = tr["legs"]
        for t in range(a, b + 1):
            realized = sum(lg["비중"] * (lg["가격"] / entry - 1) for lg in legs if lg["i"] <= t)
            left = 1.0 - sum(lg["비중"] for lg in legs if lg["i"] <= t)
            eq[t] = cur * (1 + realized + left * (close[t] / entry - 1))
        cur *= 1 + tr["ret"]; tr["equity"] = cur
        eq[b] = cur; prev_end = b
    eq[prev_end:] = cur
    equity = pd.Series(eq, index=f.index)
    s_i = cps[0] if cps else 0; e_i = min(len(f) - 1, (cps[-1] if cps else 0) + max(1, step))
    if trades:
        e_i = max(e_i, trades[-1]["exit_i"])
    rets = np.array([t["ret"] for t in trades]); Rs = np.array([t["R"] for t in trades])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    seg = equity.iloc[s_i: e_i + 1]
    held = sum(t["bars"] + 1 for t in trades)
    skipped = [d for d in decisions if not d["filled"]]
    stats = {"총 수익률": float(cur - 1), "단순 보유": float(close[e_i] / close[s_i] - 1) if e_i > s_i else 0.0,
             "MDD": float((seg / seg.cummax() - 1).min()) if len(seg) else 0.0, "거래 수": len(trades),
             "승률": float((rets > 0).mean()) if len(rets) else float("nan"),
             "손익비(PF)": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else (float("inf") if len(wins) else float("nan")),
             "평균 R": float(Rs.mean()) if len(Rs) else float("nan"), "거래당 평균": float(rets.mean()) if len(rets) else float("nan"),
             "평균 보유 봉": float(np.mean([t["bars"] for t in trades])) if trades else float("nan"),
             "시장 노출": held / max(1, e_i - s_i + 1), "판단 시점 수": len(cps), "판단 수": len(decisions), "쉰 판단": len(skipped),
             "호출 수": calls, "비용($)": round(cost, 4), "청산 사유": pd.Series([lg["사유"] for t in trades for lg in t["legs"]]).value_counts().to_dict() if trades else {},
             "기간": (f.index[s_i], f.index[e_i]), "중단": stopped}
    return {"frame": f, "equity": equity, "trades": trades, "decisions": decisions, "stats": stats, "checkpoints": cps, "range": (s_i, e_i)}


# ---------------------------------------------------------------------------
# 한 번에 훑기 (scan) — 구간을 통째로 보여 주고 패턴 자리를 한꺼번에 찍게 한다
# ---------------------------------------------------------------------------
def run_scan(bars: pd.DataFrame, spec: ChartSpec, cfg: BtConfig, *, start, end, ask_scan: Callable[[Packet, str, str], tuple[ScanResult, dict]],
             context: pd.DataFrame | None = None, ticker: str = "", window: int = 300, lead: int = 60, use_ai_exit: bool = True,
             progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """기간 [start, end] 를 ``window`` 봉씩 잘라 창마다 한 번 호출한다. 모델은 **창 전체(뒤 봉 포함)** 를 보고 자리를 고르므로 사후 판독이다 —
    결과는 "AI 눈에 이 패턴이 어디서 보이나"이지 실전 성적이 아니다. 창 앞 ``lead`` 봉은 맥락용(신호 금지).
    진입 = 신호 봉 다음 봉 시가, 청산 = 손절·목표(기계적) + (use_ai_exit) 모델이 찍은 청산 봉 종가 + 장 마감·최대 보유. 한 번에 한 포지션(겹치는 신호는 건너뜀)."""
    f = bars[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()
    F = f[["Open", "High", "Low", "Close"]]
    idx = f.index
    lo_t, hi_t = pd.Timestamp(start), pd.Timestamp(end)
    if idx.tz is not None:
        lo_t = lo_t.tz_localize(idx.tz) if lo_t.tzinfo is None else lo_t.tz_convert(idx.tz)
        hi_t = hi_t.tz_localize(idx.tz) if hi_t.tzinfo is None else hi_t.tz_convert(idx.tz)
    inside = np.flatnonzero((idx >= lo_t) & (idx <= hi_t))
    if not len(inside):
        raise ValueError("기간 안에 봉이 없습니다")
    a0, b0 = int(inside[0]), int(inside[-1])
    flat_last = _flat_last(f, spec, cfg)
    signals: list[dict] = []; overviews: list[str] = []
    calls = 0; cost = 0.0; stopped = ""
    starts = list(range(a0, b0 + 1, window))
    for n, a in enumerate(starts):
        if calls >= cfg.max_calls:
            stopped = f"호출 상한 {cfg.max_calls}회에 닿아 {idx[a]} 에서 멈춤"
            break
        b = min(b0, a + window - 1)
        if b - a < 3 and n > 0:
            break
        v0 = max(0, a - lead)
        if progress:
            progress(n / len(starts), f"구간 {n + 1}/{len(starts)} 훑는 중 · 지금까지 자리 {len(signals)}개")
        wspec = replace(spec, view_bars=b - v0 + 1, table_bars=b - v0 + 1)
        pk = build_packet(f.iloc[: b + 1], idx[b], spec=wspec, context=context, ticker=ticker, blind=cfg.blind)
        off = (b + 1) - len(pk.labels)                         # labels[k] ↔ 봉 위치 off + k
        first_lab, last_lab = pk.labels[a - off], pk.labels[-1]
        ans, u = ask_scan(pk, first_lab, last_lab)
        calls += 0 if u.get("cached") else 1; cost += u.get("cost_usd", 0.0)
        ans = ans.to_real(pk.scale, text_ref=pk.last)
        overviews.append(ans.overview)
        pos = {lab: off + k for k, lab in enumerate(pk.labels)}
        for sg in ans.signals:
            i = pos.get(sg.bar.strip())
            if i is None or i < a or i > b:
                continue                                         # 표에 없는 봉 이름이거나 맥락 구간 → 버림
            x_i = pos.get((sg.exit_bar or "").strip())
            signals.append({"i": i, "exit_i_ai": x_i if (x_i is not None and x_i > i) else None, "sig": sg, "usage": u})
    signals.sort(key=lambda d: d["i"])

    decisions: list[dict] = []; trades: list[dict] = []; busy_until = -1
    for d in signals:
        sg, i = d["sig"], d["i"]
        j = Judgement(decision="enter_long", setup=sg.setup, confidence=sg.confidence, entry_type="market", stop=sg.stop, stop_basis=sg.stop_basis,
                      target1=sg.target1, target2=sg.target2, levels=sg.levels, market_read="", structure_read=sg.note, indicator_read="", volume_read="",
                      risks="", wait_for=sg.exit_reason, summary=sg.note)
        rec = {"i": i, "time": idx[i], "close": float(f["Close"].iloc[i]), "judgement": j, "usage": d["usage"], "filled": False, "why": ""}
        decisions.append(rec)
        if i < busy_until:
            rec["why"] = "앞 거래 보유 중 (건너뜀)"
            continue
        fill = find_fill(j, F, i + 1, 1)
        if not fill["filled"]:
            rec["why"] = fill["why"]
            continue
        x_ai = d["exit_i_ai"] if use_ai_exit else None
        Fw = F.iloc[: x_ai + 1] if x_ai is not None else F
        r = walk_position(Fw, fill, j, fee_bps=cfg.fee_bps, max_hold=cfg.max_hold, flat_last=flat_last)
        if x_ai is not None and r["legs"][-1]["i"] == x_ai and r["legs"][-1]["사유"] in ("기간 끝", "시간 청산"):
            r["legs"][-1]["사유"] = "AI 청산"; r["why"] = " → ".join(lg["사유"] for lg in r["legs"])
        r.update({"decision_i": i, "decision_time": idx[i], "setup": sg.setup, "confidence": sg.confidence, "judgement": j, "exit_reason": sg.exit_reason})
        trades.append(r); rec["filled"] = True; rec["why"] = r["why"]; rec["ret"] = r["ret"]
        busy_until = r["exit_i"]
    out = _finish(f, trades, decisions, [a0, b0], max(1, b0 - a0), calls, cost, stopped)
    out["range"] = (a0, max(b0, trades[-1]["exit_i"]) if trades else b0)
    out["stats"]["기간"] = (idx[a0], idx[out["range"][1]])
    out["stats"]["단순 보유"] = float(f["Close"].iloc[out["range"][1]] / f["Close"].iloc[a0] - 1)
    out["stats"]["판단 시점 수"] = len(starts); out["stats"]["판단 수"] = len(decisions)
    out["overview"] = "\n\n".join(o for o in overviews if o); out["mode"] = "scan"
    return out
