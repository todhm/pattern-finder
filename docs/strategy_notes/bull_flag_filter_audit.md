# Bull Flag 필터 출처 감사 — 영상 명시 vs 애매 vs 백테스트 추가

`BullFlagDetector` / `BullFlagStrategy` 에 적용된 모든 필터/파라미터를 **영상에서 확정적으로 도출 가능한지** 기준으로 3분류:

| 분류 | 의미 |
|---|---|
| 🟢 **확정** | 영상에서 수치/조건이 명시적으로 언급됨 — 옮길 때 해석 여지 없음 |
| 🟡 **애매** | 영상은 *방향*은 언급하지만 정확한 수치/임계값은 안 줌 — backtest tuning 영역 |
| 🔵 **백테스트 추가** | 영상 외 — 데이터 품질 / 시뮬 정확도 보강용 가드 (Ross 영상엔 무관) |

각 항목 옆에 영상 출처(인용/시각)와 코드 default 값 병기.

---

## 1. Stock Selection (4 criteria)

| 파라미터 | 분류 | Default | 영상 근거 |
|---|---|---|---|
| `min_gap_pct` | 🟡 애매 | 2% | "최소 +2%, 이상적 +10%" — 두 수치 모두 등장. backtest는 lower bound 채택 (sample size). 영상은 "+10% 이상이 sweet spot" 강조하지만 strict +10%면 후보 거의 0건 |
| `min_rvol` | 🟢 **확정** | 5.0× | 영상 화이트보드 5 criteria 슬라이드에 "5x Relative Volume" 명시 |
| `min_price` / `max_price` | 🟢 **확정** | $2 / $20 | "주가 $2~$20" 명시 (retail 자금 매칭) |
| `max_float_shares` | 🟡 애매 | 10M | DP4 영상에서는 **20M**으로 갱신 언급, m5zu 영상은 **10M**. 두 영상이 다름 → backtest는 보수적 10M default |
| `require_float_filter` | 🔵 추가 | True | 영상은 float 필터를 *반드시* 적용. yfinance가 float 데이터 못 줄 때 어떻게 할지에 대한 코드 옵션 |

---

## 2. Pole Geometry

| 파라미터 | 분류 | Default | 영상 근거 |
|---|---|---|---|
| `pole_lookback` | 🟡 애매 | 7 봉 | 영상은 "5~7개의 그린 캔들" 표현. 봉 수 vs 시간 lookback 매핑은 해석 영역 |
| `pole_min_pct` | 🟡 애매 | 4% | 영상은 "5%, 8%, 10%" 등 다양한 예시만 등장, 명시적 임계값 X. 4%는 데이터 sparse 보정용 lower bound |
| `pole_min_green_bars` | 🟢 **확정** | 3 | 영상의 "5~7 candles" 는 패턴 전체(pole+flag+breakout) 합산 — 폴 자체는 3 green이 영상 도식과 일치. DP4 슬라이드 명시 |

---

## 3. Flag (Pullback) Geometry

| 파라미터 | 분류 | Default | 영상 근거 |
|---|---|---|---|
| `flag_max_bars` | 🟢 **확정** | 4 (영상 = 3) | 영상 명시 "1~3개의 풀백 캔들". Backtest default 4는 lenient 보정 (영상 정통은 3). |
| `flag_max_retrace` | 🟢 **확정** (영상 50%) / 🟡 (default 70%) | 70% | 영상 룰 "50% retrace 안쪽" 명시. 그러나 1m sparse 데이터에서 50%면 매칭 거의 0 — backtest default 70%로 lenient |

---

## 4. Volume Profile (DP4 영상 P0)

| 파라미터 | 분류 | Default | 영상 근거 |
|---|---|---|---|
| `enable_volume_profile` | 🟢 **확정 (방향)** | True | 영상 핵심 룰: "polymerase volume on pole, **light volume on red candles**, fresh round of buyers on breakout" |
| `pullback_volume_ratio` | 🟡 애매 | 0.7 | 영상은 "light volume on red" 정성적 표현만, 수치 임계값 명시 X. 0.7 = "flag 평균 ≤ 폴 평균의 70%" 해석 |
| `breakout_volume_ratio` | 🔵 추가 | 0.0 (off) | 영상은 "even higher volume" 정성 표현만. 수치 임계값 명시 X — backtest는 default off, opt-in 가능 |

---

## 5. Topping Tail (DP4 영상 P1)

| 파라미터 | 분류 | Default | 영상 근거 |
|---|---|---|---|
| `max_pole_topping_tail_ratio` | 🟢 **확정 (방향)** / 🟡 (수치) | 0.5 | 영상은 "긴 위꼬리 = 매수 압력 거부" 명시 (candle anatomy 화이트보드). 0.5 임계값(상위 꼬리/봉 전체)는 임의 — 영상 미명시 |

---

## 6. Multi-Timeframe Alignment (DP4 영상 33:30~35:00)

| 파라미터 | 분류 | Default | 영상 근거 |
|---|---|---|---|
| `enable_mtf_check` | 🟢 **확정** | True | 영상 VVPR 케이스 "both 1m and 5m giving the same signal" 명시 |
| `mtf_tolerance_seconds` | 🟡 애매 | 600s (10분) | 영상은 정확한 초/분 tolerance 절대 명시 X. Ross의 VVPR 발언("5m이 한 봉 앞서 가는 중")이 5~10분 거리에 해당 → 600s default 합리화 |

---

## 7. Entry / Stop / Target (영상 §3.5 정통)

| 파라미터/룰 | 분류 | 영상 근거 |
|---|---|---|
| Entry = pole_end_high | 🟢 **확정** | "first candle to make a new high" — 폴 고점 돌파 첫 봉이 트리거 |
| Stop = flag_low | 🟢 **확정** | "the low of the pullback is your max loss" |
| Target = HoD 또는 entry+2R | 🟢 **확정** (방향) / 🟡 (R-multiple은 simplification) | "first target = retest of HoD". Fixed R-multiple은 backtest simplification (영상은 HoD touch 후 더 holding) |
| `target_min_r_multiple` (2.0) | 🟢 **확정** | "I won't take the trade if I don't think I can get 2:1" |
| `latest_entry_local` (12:00 ET) | 🟢 **확정 (방향)** / 🟡 (수치) | 영상 sweet spot = "개장 후 1~2시간" (= 09:30 + 2h = **11:30**). Backtest default 12:00은 lenient |

---

## 8. Add-to-Winner (Doubling)

| 파라미터/룰 | 분류 | Default | 영상 근거 |
|---|---|---|---|
| `enable_add_to_winner` | 🟢 **확정** | True | 영상 "double the position at +20¢" 명시 |
| `add_at_r` | 🟡 애매 | 1.5R | 영상 "+20¢" 정도. backtest는 R-multiple로 normalize (가격 무관 비교 위해) — default 1.5R |
| BE stop = 최초 entry 가격 | 🟢 **확정** | — | "stop at original entry — worst case I'm flat on initial size" |
| `be_stop_buffer_pct` | 🔵 추가 | 0.3% | BE stop이 최초 entry와 정확히 같으면 진입가 한 번 tag로 즉시 발화 → 백테스트 정확도 보정 (영상엔 무관) |
| `add_confirm_on_close` | 🔵 추가 | True | 영상 명시 X. backtest에서 1m wick 한 번에 fake-add 트리거 차단용 |

---

## 9. Risk / Position Sizing

| 파라미터 | 분류 | Default | 영상 근거 |
|---|---|---|---|
| `risk_per_trade` | 🟡 애매 | 2% | 영상은 "quarter-cushion" rule (가용 자본의 1/4 사용) 정도 — 수치 명시 X |
| `max_position_pct_of_equity` | 🔵 추가 | 30% | 영상 미명시. 단일 trade notional cap. 데이트레이드 leverage 고려 |
| `max_session_losses` | 🟢 **확정 (방향)** / 🟡 (수치) | 1 | 영상 "loss 발생 → size down → 30분 미체결 시 quit" — "1회 손실 시 진입 차단"은 backtest 단순화 |

---

## 10. Second-Pullback Gate

| 룰 | 분류 | 영상 근거 |
|---|---|---|
| `pole_start_ts > last_exit_ts` | 🟢 **확정 (방향)** / 🔵 (구현) | 영상 "first and second pullback work very well" — 두 번째는 NEW pole + NEW pullback 의미. 시각 비교는 backtest 구현 디테일 |

---

## 11. Reverse-Split / Continuity Guard

| 파라미터 | 분류 | Default | 비고 |
|---|---|---|---|
| `splits` / `split_blackout_days` | 🔵 추가 | 30일 | 영상 무관. yfinance split-adjusted 데이터의 가격 불연속 false-fire 차단 (UGRO 케이스) |
| `price_floor_lookback_days` | 🔵 추가 | 30일 | 영상 무관. \$0.41 → \$7.35 점프 같은 split 흔적 catch |
| `min_bar_range` | 🔵 추가 | $0.001 | 영상 무관. yfinance zero-range "doji" 봉 (무거래 시간대) reject — Ross 화면엔 dense 분봉만 등장 |
| `max_bar_gap_seconds` | 🔵 추가 | 90s | 영상 무관. 1m 봉 간 시간 갭 (= 거래 단절) reject |

---

## 12. 수수료

| 파라미터 | 분류 | Default | 비고 |
|---|---|---|---|
| `TossFeeSchedule` (buy 0.1% + sell 0.1% + SEC 0.0023%) | 🔵 추가 | — | 영상은 미국 broker (commission-free) 가정. 한국 사용자 토스증권 수수료 시뮬용 |

---

## 분류별 요약

### 🟢 영상 확정 (수치/조건 명시) — 옮길 때 해석 여지 없음
- min_rvol = **5.0×**
- min_price / max_price = **$2 / $20**
- pole_min_green_bars = **3**
- flag_max_bars = **3** (코드 default 4는 lenient)
- flag_max_retrace = **50%** (코드 default 70%는 lenient)
- enable_volume_profile (방향)
- enable_mtf_check (방향)
- Entry / Stop / Target 룰 (방향)
- target_min_r_multiple = **2.0**
- enable_add_to_winner (방향)
- BE stop = 최초 entry (방향)

### 🟡 애매 (영상이 방향만, 수치는 backtest tuning)
- min_gap_pct, max_float_shares (영상마다 수치 다름)
- pole_lookback, pole_min_pct
- pullback_volume_ratio
- max_pole_topping_tail_ratio (수치)
- mtf_tolerance_seconds (영상 명시 X)
- target_at_r_multiple (영상은 HoD, R-multiple은 simplification)
- latest_entry_local (영상 11:30, default 12:00)
- add_at_r (영상 +20¢, R로 normalize)
- risk_per_trade (영상 "quarter-cushion" 정성)
- max_session_losses (수치)

### 🔵 백테스트 추가 (영상 무관, 데이터 품질 / 시뮬 보정)
- breakout_volume_ratio (영상 정성, 수치 X — opt-in)
- be_stop_buffer_pct, add_confirm_on_close (시뮬 정확도)
- max_position_pct_of_equity (안전 cap)
- splits / split_blackout_days (split 데이터 가드)
- price_floor_lookback_days (가격 점프 catch)
- min_bar_range, max_bar_gap_seconds (zero-range / 데이터 갭 가드)
- TossFeeSchedule (한국 broker 수수료)
- require_float_filter (float 데이터 결측 처리)

---

## 영상 정통(strict) 모드 vs 코드 default 차이

영상에 가장 충실하게 setting 하면:

| 파라미터 | 코드 default | **영상 strict** |
|---|---|---|
| flag_max_bars | 4 | **3** |
| flag_max_retrace | 70% | **50%** |
| latest_entry_local | 12:00 | **11:30** |
| min_gap_pct | 2% | **10%** |
| max_float_shares | 10M | **10M** (m5zu) 또는 **20M** (DP4) |

영상 strict 모드로 돌리면 매칭 신호 수 급감 — Ross의 "셋업 자체가 매우 드물다 (하루 한 손가락)" 발언과 일관됨. Backtest default는 sample size 확보를 위한 lenient 셋팅.
