# Docs

| 디렉토리 / 파일 | 내용 |
|---|---|
| [`architecture/ports-and-adapters.md`](./architecture/ports-and-adapters.md) | 헥사고날 구조 규칙 + 현재 포트/어댑터 매핑 |
| [`architecture/signals.md`](./architecture/signals.md) | Signals 모듈 (BuySignal watchlist) 상세 |
| [`db/alembic.md`](./db/alembic.md) | SQLAlchemy + Alembic 사용 가이드 |
| [`strategy/wedgepop/README.md`](./strategy/wedgepop/README.md) | Wedge Pop 패턴 + 전략 규칙 |
| [`strategy/wick_play/sweep_2026_04.md`](./strategy/wick_play/sweep_2026_04.md) | Wick Play 파라미터 스윕 기록 |
| [`strategy_notes/ross_cameron_bull_flag_deep_dive.md`](./strategy_notes/ross_cameron_bull_flag_deep_dive.md) | Bull Flag 전략 리서치 노트 |
| [`strategy_notes/ross_cameron_million_in_51days.md`](./strategy_notes/ross_cameron_million_in_51days.md) | Ross Cameron 51일 챌린지 노트 |
| [`strategy_notes/bull_flag_filter_audit.md`](./strategy_notes/bull_flag_filter_audit.md) | Bull Flag 필터 게이트 감사 |
| [`strategy_notes/bull_flag_video_research_2026_05.md`](./strategy_notes/bull_flag_video_research_2026_05.md) | Bull Flag YouTube 2편 정리 (Chart Mill + Tesla scalper) |
| [`strategy_notes/라이프사이클_트레이드_쉬운설명_2026_10.md`](./strategy_notes/라이프사이클_트레이드_쉬운설명_2026_10.md) | «The Lifecycle Trade» 초보자용 그림 설명 — 양봉·음봉만 알아도 따라가는 단계별 가이드 |
| [`strategy_notes/라이프사이클_트레이드_체크리스트_2026_10.md`](./strategy_notes/라이프사이클_트레이드_체크리스트_2026_10.md) | «The Lifecycle Trade» IPO 생애주기 체크리스트 (자동/수동 구분) + 검증 절차 |
| [`strategy_notes/라이프사이클_트레이드_검증_2026-10.md`](./strategy_notes/라이프사이클_트레이드_검증_2026-10.md) | IPO 코호트로 6장 통계·5장 매도 규칙 재현 + **$100K 계좌로 그대로 돌리면 얼마가 되나**(시나리오별 CAGR·MDD, SPY 대조) |
| [`strategy_notes/미너비니_WedgePop거래28건_2025-05~2026-04_수동검증.md`](./strategy_notes/미너비니_WedgePop거래28건_2025-05~2026-04_수동검증.md) | WedgePop 거래 로그 28건을 미너비니 체크리스트(진입 전날 기준)로 재판정 — 매수 후보 0~1건(LITE), 통과 여부와 손익 무관 |
| [`strategy_notes/미너비니_2026-10-13_스크리너_57종목_수동검증.md`](./strategy_notes/미너비니_2026-10-13_스크리너_57종목_수동검증.md) | 44 스크리너 10/13 내보내기 57종목을 STAGE 0~4 자동 판정 + 차트 눈판독 — 0-3 실패로 매수 0, 워치 UFCS·ZBRA·CLMT. 같은 형식의 10/09·10/10 판독과 짝 |
| [미너비니_2026-10-18_스크리너_38종목_수동검증.md](strategy_notes/미너비니_2026-10-18_스크리너_38종목_수동검증.md) | 10/18 스크리너 내보내기 38종목을 10/16 종가로 STAGE 0~4 재판정 + 눈판독. 시장이 조정 진입(QQQ·SPY 50일선 이탈, 0-3 312/586) → 신규 매수 없음. UFCS 는 탈락이 아니라 1순위 워치 유지(10/13 돌파 실패 뒤 저거래량 되밀림, 11/3 실적 전 판단), 2순위 CLMT, 3순위 CHEF(10/28 실적 후). DDOG·FROG·NTAP −14~15% 로 '추격 금지' 판정 검증 |
| [TQQQ_초과수익_로드맵_2026-10.md](strategy_notes/TQQQ_초과수익_로드맵_2026-10.md) | **유일한 로드맵 (10/19).** 세 기둥 프레임 — 1 종목선정(바스켓 μ/σ² 점수판, BAB·듀얼 모멘텀), 2 타이밍(조기경보 혼동행렬: 추세·레짐·거시·VIX·폭), 3 기타(켈리·레버리지 구조·세금을 비용으로 넣은 동적 최적화·검증). 결합 공식 w = f·μ̂/σ̂², 통과 기준 A·B·C + 과적합 보정, 가설 카드·재현 관문, 20주 일정, 기각된 가설 장부 |
| [tqqq_optimal_leverage_2026-10.md](strategy_notes/tqqq_optimal_leverage_2026-10.md) | Phase 0 결과: ^NDX 1986~2026 연도별 μ·σ·L\*. 2010~2026 L\*=3.95(3배가 최적), 2000~2009 −0.27, 전체 2.01. 참사 5년(σ≥32%)이 3배 보유 손익을 결정, 단 σ≥30% 8년 중 3년은 폭발 → Phase 1 의 교환비 |
| [변동성타기팅_레짐_검증_2026-10.md](strategy_notes/변동성타기팅_레짐_검증_2026-10.md) | Phase 1 결과: 변동성 타기팅(목표 30~120%, 예측기 4종) × ⑩ 낙폭 레짐, 세 표본·10년 롤링·세후. 22% 에서 §5 미통과(⑩ 0.93/−67%, vt75+⑩ 0.78/−62%), 9.9% 면 vt75×rv20×⑩ 이 A·C 통과. 스위치가 낙폭, 다이얼이 최악 10년을 막는다 → Phase 3(세금) 앞당김 |
| [변동성타기팅_레짐_쉬운설명_2026_10.md](strategy_notes/변동성타기팅_레짐_쉬운설명_2026_10.md) | Phase 1 쉬운 설명판: 이 검증이 무엇을 하려는 건지(브레이크 두 개: 스위치와 다이얼), 그림 3장(브레이크 작동 구간, 교환비 곡선, 10년 롤링), 결과를 한 문장씩 |
| [참사_조기경보_검증_2026-10.md](strategy_notes/참사_조기경보_검증_2026-10.md) | 기둥 2 첫 실험: 기간·신용 스프레드, 실업률·청구, VIX, 추세 7신호로 ⑩ 이탈을 앞당길 수 있나. 답: 못 한다 — 100~120영업일 너무 일찍 켜져 이탈 조건에 넣으면 휩소 2~3배(2010 표본 0.12~0.66), 참사 5년 평균 −45% 불변. 남는 용도는 약세장 재진입 게이트(F4) |
| [study_cards/README.md](strategy_notes/study_cards/README.md) | 가설 장부: 기각된 가설 R1~R11, 살아 있는 사실 F1~F4, 카드 C1~ |
| [`research/youtube-strategy-extraction.md`](./research/youtube-strategy-extraction.md) | YouTube 전략 추출 워크플로우 |
| [`backtest_alphafolio.md`](./backtest_alphafolio.md) | Alphafolio 3-서비스 스택의 PIT 백테스트 설계 |

프로젝트 전반 사용법은 루트의 [`README.md`](../README.md), 개발 규칙은 [`CLAUDE.md`](../CLAUDE.md) 참조.
