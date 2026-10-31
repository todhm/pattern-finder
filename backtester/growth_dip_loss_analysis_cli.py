"""성장주 당일 눌림 그리드의 거래 목록(`trades_out` json)에 진입일 특징을 붙여 이익/손실 거래를 비교한다.

    docker compose exec backtester python growth_dip_loss_analysis_cli.py <trades.json> <out.txt>

분봉 확정 12 조합 전체 + 일봉 pessimistic(지수필터 없음·PER 없음·tp2 10/60) 을 표본으로, 수치 특징(종가 위치·실적까지 일수·
거래량 배수·200일선·QQQ 당일 등)의 이익/손실/큰손실 평균과 3분위별 손실 비율, 범주별(청산 사유·레짐·연도·요일·산업군)을 출력하고
표본별 CSV(_intra.csv / _daily.csv)를 같이 남긴다. 결과 예: docs/strategy_notes/growth_dip_runs/growth_dip_loss_analysis_2026-10-31.txt"""
import json, sys, numpy as np, pandas as pd
from datetime import date
from data.adapters.yfinance_bulk import download_many
from growth_dip_scalp_cli import load_snapshots
from strategy.adapters.evidence_backtest import classify_regimes
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
TR = sys.argv[1]; OUT = sys.argv[2]
d = json.load(open(TR))
def sel(kind, f):
    rows = []
    for b in d:
        if b['kind'] != kind or not f(b['rules']): continue
        for t in b['trades']:
            rows.append({**t, 'trigger': b['rules']['trigger'], 'x': b['rules']['x_pct'], 'tp2': b['rules']['tp2_pct'], 'hold_max': b['rules']['max_hold_days']})
    return pd.DataFrame(rows)
intra = sel('intraday', lambda r: True)
daily = sel('daily', lambda r: r['fill'] == 'pessimistic' and r['per_filter'] is None and not r['index_filter'] and r['tp2_pct'] == 10 and r['max_hold_days'] == 60)
print("intraday trades", len(intra), "daily(pess, idx-, per-, 10/60) trades", len(daily))
tickers = sorted(set(intra.ticker) | set(daily.ticker))
px = download_many(tickers + ['QQQ'], date(2015, 1, 1), date(2026, 10, 24))
qqq = px.pop('QQQ')
snaps, industry = load_snapshots(tickers)
earn = {t: sorted(pd.Timestamp(q.report_date) for q in qs if q.report_date) for t, qs in snaps.items()}
reg = classify_regimes(qqq); reg_by_month = {(k.year, k.month): v for k, v in reg.regime.items()}
qc = qqq['Close']; q50 = qc.rolling(50).mean()
def feats(t, day):
    df = px.get(t)
    if df is None or day not in df.index: return {}
    i = df.index.get_loc(day)
    if i < 210: return {}
    C, O, H, L, V = (df[c].to_numpy(float) for c in ('Close', 'Open', 'High', 'Low', 'Volume'))
    pc = C[i-1]
    tr = np.maximum(H[i-20:i]-L[i-20:i], np.maximum(abs(H[i-20:i]-C[i-21:i-1]), abs(L[i-20:i]-C[i-21:i-1])))
    streak = 0
    for k in range(i-1, max(i-11, 0), -1):
        if C[k] < C[k-1]: streak += 1
        else: break
    nxt = [e for e in earn.get(t, []) if e > day]; prv = [e for e in earn.get(t, []) if e <= day]
    return dict(gap=O[i]/pc-1, day_ret=C[i]/pc-1, low_depth=L[i]/pc-1, close_pos=(C[i]-L[i])/(H[i]-L[i]) if H[i] > L[i] else np.nan,
                ret5=pc/C[i-6]-1, ret20=pc/C[i-21]-1, ret60=pc/C[i-61]-1, pos_sma20=pc/C[i-20:i].mean()-1, pos_sma50=pc/C[i-50:i].mean()-1,
                pos_sma200=pc/C[i-200:i].mean()-1, off_high60=pc/C[i-60:i].max()-1, vol_ratio=V[i]/V[i-20:i].mean() if V[i-20:i].mean() > 0 else np.nan,
                atr_pct=tr.mean()/pc, down_streak=streak, qqq_day=float(qc.loc[day]/qc.shift(1).loc[day]-1), qqq_pos50=float(qc.loc[day]/q50.loc[day]-1),
                days_to_earn=(nxt[0]-day).days if nxt else np.nan, days_since_earn=(day-prv[-1]).days if prv else np.nan,
                weekday=day.weekday(), year=day.year, regime=reg_by_month.get((day.year, day.month), '?'), industry=industry.get(t) or '?',
                price=pc)
def enrich(T):
    T = T.copy(); T['day'] = pd.to_datetime(T['day'])
    F = pd.DataFrame([feats(t, dd) for t, dd in zip(T.ticker, T.day)], index=T.index)
    T = pd.concat([T, F], axis=1); T['win'] = T.ret > 0; T['big_loss'] = T.ret < -0.05
    return T
num = ['gap', 'day_ret', 'low_depth', 'close_pos', 'ret5', 'ret20', 'ret60', 'pos_sma20', 'pos_sma50', 'pos_sma200', 'off_high60', 'vol_ratio', 'atr_pct', 'down_streak', 'qqq_day', 'qqq_pos50', 'days_to_earn', 'days_since_earn', 'price', 'hold']
cat = ['why', 'regime', 'year', 'weekday', 'trigger', 'x', 'tp2', 'industry']
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)
for name, T in (("5분봉 확정 12 조합 (표본 12 달)", enrich(intra)), ("일봉 pessimistic · 지수필터 없음 · PER 없음 · tp2 10/60 (트리거 2 × X 3)", enrich(daily))):
    P(f"\n######## {name}: n={len(T)} 승률 {T.win.mean():.0%} 거래당 {T.ret.mean():+.2%} 큰손실(<-5%) {T.big_loss.mean():.0%} · 손실 거래 합 {T.ret[T.ret<0].sum():+.1f} vs 이익 합 {T.ret[T.ret>0].sum():+.1f} (거래 수 비율 손실 {(~T.win).mean():.0%})")
    P("\n## 수치 특징: 이익 거래 vs 손실 거래 vs 큰 손실(<-5%) 평균 (중앙값)")
    P("| 특징 | 이익 | 손실 | 큰 손실 | 손실-이익 |")
    for c in num:
        w, l, b = T.loc[T.win, c], T.loc[~T.win, c], T.loc[T.big_loss, c]
        P(f"| {c} | {w.mean():.3f} ({w.median():.3f}) | {l.mean():.3f} ({l.median():.3f}) | {b.mean():.3f} ({b.median():.3f}) | {l.mean()-w.mean():+.3f} |")
    P("\n## 수치 특징 3분위별 손실 비율 · 거래당 (낮음/중간/높음)")
    for c in num:
        try:
            q = pd.qcut(T[c], 3, labels=['낮음', '중간', '높음'], duplicates='drop')
        except Exception: continue
        g = T.groupby(q, observed=True).agg(n=('ret', 'size'), loss=('win', lambda s: 1-s.mean()), big=('big_loss', 'mean'), ret=('ret', 'mean'))
        edges = T[c].quantile([1/3, 2/3]).round(3).tolist()
        P(f"- {c} (경계 {edges}): " + " · ".join(f"{k} n{int(r.n)} 손실 {r.loss:.0%} 큰손실 {r.big:.0%} 거래당 {r.ret:+.2%}" for k, r in g.iterrows()))
    P("\n## 범주별")
    for c in cat:
        g = T.groupby(c).agg(n=('ret', 'size'), loss=('win', lambda s: 1-s.mean()), big=('big_loss', 'mean'), ret=('ret', 'mean'), hold=('hold', 'mean')).sort_values('n', ascending=False)
        if c == 'industry': g = g[g.n >= 15].sort_values('ret')
        P(f"- {c}: " + " · ".join(f"{k} n{int(r.n)} 손실 {r.loss:.0%} 큰손실 {r.big:.0%} 거래당 {r.ret:+.2%} 보유 {r.hold:.1f}" for k, r in g.iterrows()))
    P("\n## 당일 익절(hold 0) vs 보유로 넘어간 거래의 진입 당시 특징")
    h0, h1 = T[T.hold == 0], T[T.hold > 0]
    P(f"당일 익절 n{len(h0)} / 보유 n{len(h1)} (보유 거래 승률 {h1.win.mean():.0%} 거래당 {h1.ret.mean():+.2%})")
    for c in ['gap', 'day_ret', 'low_depth', 'close_pos', 'ret5', 'ret20', 'pos_sma50', 'off_high60', 'vol_ratio', 'atr_pct', 'qqq_day', 'days_to_earn']:
        P(f"  {c}: 당일익절 {h0[c].mean():.3f} / 보유 {h1[c].mean():.3f} / 보유 중 손실 {h1.loc[~h1.win, c].mean():.3f} / 보유 중 이익 {h1.loc[h1.win, c].mean():.3f}")
    P("\n## 큰 손실 상위 15")
    P(T.sort_values('ret').head(15)[['day', 'ticker', 'trigger', 'x', 'ret', 'why', 'hold', 'gap', 'day_ret', 'ret20', 'pos_sma50', 'off_high60', 'days_to_earn', 'regime', 'industry']].round(3).to_string())
    T.to_csv(OUT.replace('.txt', f"_{'intra' if 'ref' not in name and '5분봉' in name else 'daily'}.csv"), index=False)
open(OUT, 'w').write("\n".join(lines))
