"""AI 판정용 패턴 목록 — 단일 종목 패턴 페이지(5·9·10·11·13·15·16·18·19·21·23·55)마다 하나.

각 항목은 모델에게 줄 **원칙 글**(숫자 문턱이 아니라 트레이더가 말로 설명하는 패턴)과 차트 구성·기본값이다.
원칙 글은 각 규칙 엔진(pattern/adapters·strategy/adapters)의 설명을 옮긴 것이고, 엔진의 세부 파라미터는 일부러 뺐다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from strategy.adapters.ai_chart_judge import ChartSpec


@dataclass(frozen=True)
class PatternSpec:
    key: str
    title: str                     # 화면·모델에게 보이는 이름
    rule_page: str                 # 대응하는 규칙 엔진 페이지
    chart: ChartSpec
    principles: str
    source: str = "stack_rth"      # av_ext | stack_rth | stack_all | crypto | daily
    ticker: str = "AAPL"
    days: int = 5                  # 기본 기간(달력 일)
    warmup_days: int = 6           # 기간 앞에 더 읽을 달력 일 (차트·지표용)
    step: int = 4                  # 몇 봉마다 묻나
    max_hold: int = 0
    flat_at_close: bool = False
    regular_only: bool = False
    ask_from: str = ""             # 진입을 묻는 시간대 (뉴욕 "HH:MM", 빈 값 = 제한 없음)
    ask_until: str = ""
    note: str = ""                 # 화면 안내 (규칙 엔진과 다른 점 등)
    extra: dict = field(default_factory=dict)


_VWAP_SR = """세션 VWAP 근처에서 만들어진 지지·저항 구조의 돌파를 15분봉으로 봅니다 (Udemy VWAP 강의 — 사용자가 화면 53장에 적어 둔 메모를 옮긴 것).
차트: 15분봉, 날마다 리셋되는 세션 VWAP(주황 선), 거래량. 롱만 봅니다(강의의 숏 사례는 방향만 뒤집은 같은 원리입니다).

[1] 지지·저항을 보는 법
- 지지·저항은 선이 아니라 "영역"입니다. 지지 = 매수자가 관심을 두는 가격대(잠재 매수 압력), 저항 = 매도자가 관심을 두는 가격대(잠재 매도 압력). 시장은 그 영역을 조금 넘거나 못 미칩니다.
- 많이 시험받을수록 약해집니다. 저항을 계속 두드리면 거기 쌓인 매도 주문이 체결되어 사라지고, 그러면 더는 막을 사람이 없습니다.
- 저항을 향해 저점이 높아지며 올라오면(상승 삼각형) 뚫릴 때가 더 많습니다 — 매수자가 더 비싼 값에도 사겠다는 뜻이고 힘의 표시입니다.
- 짧은 시간에 여러 번 찔렀는데 "멀리 밀려나지 않고" 저항 바로 밑에 붙어 횡보하면 곧 뚫릴 가능성이 높습니다. 매도세가 장악했다면 가격을 끌어내렸을 텐데 못 했기 때문입니다.
- 깨진 지지는 저항이 되고, 뚫린 저항은 지지가 되는 경향이 있습니다.
- 저항은 상승 추세 방향으로 뚫리기 쉽습니다. 추세와 같은 방향으로 거래합니다("trend is your friend until it bends").

[2] 잡아야 하는 자리 (진입)
- 반드시 빌드업이 있어야 합니다: 저항 근처의 횡보(압축 = 에너지 저장), 상승 삼각형, 또는 내려오는 저항 추세선과 받쳐 주는 지지선 사이의 쐐기. 횡보(트레이딩 레인지)는 길수록 좋고, 좁을수록 손절을 가깝게 둘 수 있어 좋습니다.
- 저항이 VWAP 위에 있을 때 롱입니다. 가격이 VWAP 근처에서 저항을 두드리다가 봉이 저항 위로 "마감"하면 진입합니다. 진입은 VWAP 에 최대한 가까이.
- 횡보(박스권) 중에 터지는 거래량 = 추세의 시작입니다. 박스·추세선을 돌파하는 봉에 거래량이 실리면 진입 근거입니다.
- 하루에 두 번도 합니다. 청산(익절·손절) 뒤 같은 모양이 다시 나오면 재진입합니다. 특히 거래량을 동반해 VWAP·저항을 다시 넘는 봉은 재진입 자리입니다.
- 손절 사냥 뒤에 진짜 추세가 나오는 일이 많습니다. 지지 밑으로 긴 꼬리를 찍고 되돌아온 뒤의 돌파는 좋은 자리입니다.

[3] 피해야 하는 자리
- 큰 봉 하나로 뻥 뚫은 돌파: 사람들이 공포·조급함에 추격하고, 곧 되돌려지기 쉽습니다.
- 빌드업 없이 강한 추세로 곧장 달려와 한 번에 뚫은 저항: 이미 수익 중인 사람들이 저항에서 팔고 매도자도 나와 가짜 돌파가 됩니다.
- VWAP 에서 이미 너무 멀리 떨어진 자리.
- 손절을 어디 둘지 합리적으로 정할 수 없는 자리. "무엇을 거는지 모르면 전부를 거는 것"입니다.

[4] 손절
- 지지 영역 "정확히"가 아니라 조금 밖에 둡니다. 지지 바로 밑은 모두가 손절을 두는 곳이라 사냥당하기 쉽습니다.
- 박스 돌파 진입이면 박스 아랫선(지지선) 밖, 쐐기면 받쳐 주던 지지 추세선 밖. 마땅한 지지가 없으면 VWAP 아래.
- 꼬리가 손절선을 잠깐 넘는 것으로는 나가지 않습니다. 봉이 손절선 아래로 "마감"해 확정될 때만 손절합니다(stop_basis = close).

[5] 청산(익절) — 가장 틀리기 쉬운 부분이니 순서대로 따릅니다
원칙: "진입은 VWAP 가까이, 청산은 VWAP 에서 최대한 멀리". 가격이 VWAP 에서 충분히 멀어진 뒤에야 "나갈 이유"를 찾기 시작합니다.
(가) 아직 VWAP 근처일 때 — 돌파 직후, 진입 뒤 횡보, VWAP 을 타고 천천히 오르는 중 — 는 청산을 찾는 단계가 아닙니다. 이때 나가는 이유는 손절(지지/VWAP 아래 마감)뿐입니다.
   · 돌파 봉이나 그 직후에 터진 거래량은 "추세의 시작"이지 익절 신호가 아닙니다. 강의 메모: "돌파에 따른 거래량이 나오고 아직 volume spike 가 일어나지 않았으므로 VWAP 을 reject 했다고 볼 수 없다 — 익절 포인트 아님".
   · 가격이 VWAP 에서 떠나기 전에 나온 작은 음봉은 청산 신호가 아닙니다. 강의 메모: "VWAP rejection 이 일어난 다음 candle 에서 익절 — 그 전의 빨간 candle 은 VWAP rejection 이라고 볼 수 없다".
   · 진입 봉이나 바로 다음 봉에서 평범한 음봉·작은 꼬리 하나를 보고 나가지 마세요. 그것은 추세를 통째로 놓치는 가장 흔한 실수입니다.
(나) 가격이 VWAP 을 뿌리치고(VWAP rejection) 뚜렷이 멀어진 뒤 — 진입 뒤 여러 봉에 걸쳐 올라 VWAP 과의 간격이 눈에 띄게 벌어진 상태 — 에서 아래 중 처음 나오는 것을 보고 그 봉 마감에 청산합니다:
   1. 진행 중인 추세에서 터지는 거래량 스파이크(클라이맥스). 이미 달리는 추세의 거래량 급증은 추세의 끝일 가능성이 높습니다. → 그 봉 마감에 바로 청산.
   2. 멀어진 상태에서 나오는 첫 역캔들(첫 음봉) — "방향이 바뀌었다".
   3. 긴 윗꼬리 — 위에서 매도자가 나타났다는 증거.
   4. 커다란 봉이 나와 점프한 뒤 — 큰 봉은 되돌려지는 일이 많으니 그 봉이 끝나면 청산.
   ※ "충분히 멀어졌다"의 감(강의에 숫자는 없고 이 도구가 정한 기준): 종가와 VWAP 의 간격이 요약의 ATR 의 2배쯤 이상이고, 평가 수익이 진입가–손절가 거리(1R)를 넘었을 때.
     수익이 1R 도 안 되는데 음봉·꼬리가 나왔다고 나가면 손실은 1R 씩 내고 이익은 조금씩만 얻게 됩니다. 그 상태에서는 (다)의 뚜렷한 증거가 없는 한 계속 보유합니다.
(다) 예외적으로 빠른 청산: 진입 직후라도 "거래량 스파이크와 함께" 급하게 VWAP 에서 튀어 올랐거나, 매도자가 많다는 뚜렷한 증거(거래량이 실린 긴 윗꼬리)가 보이면 다음 봉을 기다리지 않고 그 봉 마감에 나갑니다. 거래량도 꼬리도 평범하면 이 예외가 아닙니다.
(라) 고정 목표가는 쓰지 않습니다. target1·target2 는 null 로 두고, 청산은 위 신호(exit_bar 또는 보유 중 판단)로만 합니다.

[6] 큰 그림과 시간대
- 일봉 추세, 전일 고가·저가·종가, 장전 움직임을 함께 봅니다.
- 미국 주식의 시간외(회색 배경) 봉은 거래가 얇아 꼬리와 거래량을 덜 믿습니다. 시간외의 작은 거래량 변화는 스파이크가 아닙니다 — 거래량 스파이크는 주변 정규장 봉들과 견주어 뚜렷이 큰 것만 칩니다."""

_WEDGE_POP = """Oliver Kell / TraderLion 의 "Wedge Pop"(머니 패턴) — 조정이 끝나고 다시 오르기 시작하는 첫 신호를 삽니다.
- 배경: 종목이 큰 상승 추세 안에 있습니다(장기 이동평균선 위, 중기 이동평균선이 오르는 중). 하락 추세의 반등은 이 패턴이 아닙니다.
- 모양: 고점에서 밀려 한동안 단기 이동평균선(빠른 EMA) 아래에서 좁아지며 조정합니다 — 봉이 작아지고 거래량이 마르고 저점이 더 깊어지지 않는 쐐기·횡보.
- 신호: 그 조정을 벗어나며 빠른 EMA 와 느린 EMA 를 한 번에 위로 넘겨 마감하는 힘 있는 양봉(거래량 증가가 따르면 더 좋음). 다음 봉 시가에 매수합니다.
- 손절: 조정 구간의 저점 바로 아래.
- 피할 것: 신호 봉이 이미 이동평균선에서 너무 멀리 뻗은 자리(추격), 과열된 장대 양봉, 큰 그림이 하락 추세, 조정 없이 계속 오르던 중의 그냥 양봉.
- 청산: 상승 뒤 과열 급등(이동평균선에서 크게 벌어진 장대 양봉 뒤 반전), 저점을 이은 상승 추세선 이탈, 빠른 EMA 아래 마감이 이어질 때. 1R 이상 수익이 나면 손절을 본전으로 올립니다."""

_WICK_PLAY = """Oliver Kell 의 "Wick Play" — 상승 추세 속 3봉 재돌파 패턴.
- 배경: 최근 뚜렷한 상승 추세(이동평균선 위, 정배열).
- 1봉(윅 봉): 위로 치솟았다가 밀려 윗꼬리가 길게 남은 봉(꼬리가 봉 범위의 절반 이상). 거래량이 과하지 않아야 합니다 — 대량 거래로 찍힌 윗꼬리는 분산(매도)입니다.
- 2봉(인사이드): 윅 봉 범위 안에서 조용히 쉬는 봉, 거래량이 줄어듭니다. 매도 압력이 이어지지 않았다는 뜻입니다.
- 3봉(돌파): 윅 봉의 고가 위로 마감하는 양봉, 거래량이 늘어납니다. 윗꼬리에서 판 사람들의 물량을 다 받아 냈다는 신호 — 다음 봉 시가에 매수합니다.
- 손절: 윅 봉의 저가 아래.
- 피할 것: 연속 음봉 중의 윗꼬리, 상승 추세가 없는 자리, 윅 봉 범위가 지나치게 큰 경우, 큰 그림이 약세, 갭 하락으로 열리는 진입.
- 청산: 빠른 이동평균선(단기 EMA) 아래로 마감하면 추세가 꺾인 것 — 계단식으로 따라 올립니다. 과열 급등 뒤 반전도 청산 신호입니다."""

_FVG = """Fair Value Gap(공정가치 갭) 되돌림 매수 — 스마트머니 개념의 추세 전환 진입.
- 1단계 — 성격 변화(CHoCH): 저점을 낮추며 내려오던 가격이, 마지막 하락의 스윙 고점 위로 "마감"하며 구조를 뒤집습니다. 이게 없으면 자리가 아닙니다.
- 2단계 — FVG: CHoCH 뒤 힘 있게 오르는 3봉에서 첫 봉의 고가와 셋째 봉의 저가 사이에 겹치지 않는 빈 구간(갭)이 생깁니다. 가운데 봉이 큰 양봉(변위)입니다. 갭이 너무 작으면 의미가 없습니다.
- 진입: 추격하지 않습니다. 가격이 그 갭으로 되돌아올 때 갭 중간쯤에 지정가(limit)로 삽니다. 갭이 생긴 직후 몇 봉 안에 되돌림이 와야 하고, 너무 늦게 오거나 갭을 완전히 메우고 내려가면 무효입니다.
- 손절: 갭을 만든 가운데 봉의 저가 아래.
- 목표: 위험의 약 3배 또는 위쪽 유동성(직전 고점). 진입 뒤 새 스윙 고점 위로 마감(BOS)하면 손절을 본전 근처로 올립니다.
- 피할 것: CHoCH 없는 갭, 하락 추세 한가운데의 갭, 이미 한 번 채워진 갭."""

_FVG_US = _FVG + """
- 미국 주식: 하루 세션 단위로 봅니다. 장 마감까지 시간이 얼마 안 남았으면 진입하지 않고, 포지션은 장 마감에 정리됩니다. 시간외 봉은 거래가 얇아 덜 믿습니다."""

_FVG_BTC = _FVG + """
- 비트코인: 24시간 거래라 세션 마감이 없습니다. 구조는 날을 넘어 이어집니다. 주말·새벽의 얇은 시간대 갭은 덜 믿습니다."""

_SCRAFACE = """Scraface 시가 범위(ORB) 재시험 — 눌려 있던 종목의 반등을 장 초반에 삽니다 (1분봉).
- 배경(일봉): 최근 대부분의 날을 50일 이동평균선 아래에서 마감한 눌린 종목. 그 억눌림에서 튀는 날을 노립니다.
- 박스: 정규장 첫 5분봉의 고가·저가. 요약의 '오늘 첫 5분 고가/저가'가 그 박스입니다. 박스 안에서는 거래하지 않습니다.
- 순서: ① 박스 고가 위로 마감하는 돌파 → ② 다시 박스 고가 아래로 마감하는 되돌림(재시험) → ③ 그다음 박스 고가 위로 다시 마감하는 봉이 진입입니다. 첫 돌파를 바로 사지 않습니다.
- 손절: 진입 봉의 저가 아래. 목표: 위험의 약 2배.
- 장 초반(대략 11시 전)에만 진입하고, 포지션은 장 마감 전에 정리합니다.
- 피할 것: 재시험 없이 달아난 돌파 추격, 박스가 지나치게 넓은 날, 재시험이 박스 깊숙이 무너진 경우."""

_TRADESHARP = """Trade Sharp ORB — 15분 시가 범위의 '함정 뒤' 진입 (5분봉).
- 배경: 전일 일봉이 양봉(강세 편향)인 날만 롱을 봅니다.
- 박스: 정규장 첫 15분의 고가·저가(요약의 '오늘 첫 15분 고가/저가').
- 첫 돌파는 함정으로 봅니다. 박스 위로 한 번 나갔다가 다시 범위 쪽으로 끌려 들어오는 되돌림(유동성 사냥)을 기다립니다.
- 되돌림 뒤 몇 봉(3봉 이상) 좁게 다지고, 아래꼬리가 긴 강세 거부 캔들이 나오면 준비 — 다음 봉이 그 거부 캔들의 고가를 넘으면 진입합니다(그 고가 위 마감을 조건으로 적으세요).
- 손절: 거부 캔들의 저가 아래. 목표: 위험의 약 1.5배.
- 대략 11시 30분 전에만 진입하고, 장 마감 전에 정리합니다.
- 피할 것: 첫 돌파 추격, 다지기 없는 V자 반등, 전일이 약세 봉인 날."""

_FIRST_CANDLE = """First Candle Rule (Casper SMC) — 첫 5분봉 범위와 1분봉 FVG (1분봉).
- 박스: 정규장 첫 5분봉의 고가·저가(요약의 '오늘 첫 5분 고가/저가'). 그 뒤 1분봉으로 봅니다.
- 경로 A: 박스 고가를 뚫는 봉 자체가 강한 변위 봉이라 3봉 FVG(첫 봉 고가와 셋째 봉 저가 사이의 빈 구간)를 만들면, 그 갭이 확정되는 봉의 마감에 바로 진입합니다.
- 경로 B: 돌파에 FVG 가 없었다면 박스 쪽으로 되돌아오는 재시험을 기다렸다가, 그 뒤에 나오는 다음 강세 FVG 에서 진입합니다.
- 손절: FVG 첫 봉의 저가 바로 아래. 꼬리로 살짝 찍는 것에는 관대하게 — 봉 마감 기준으로 봅니다. 목표: 위험의 약 2배. 한 시간쯤 지나도 못 가면 본전으로 올립니다.
- 피할 것: 갭 상승이 너무 큰 날(이미 달아남), 거래량이 평소보다 눈에 띄게 적은 날, 11시 이후의 진입. 포지션은 장 마감 전에 정리합니다."""

_BULL_FLAG = """Ross Cameron 의 Bull Flag — 장 초반 급등주의 깃발 눌림 (1분봉).
- 종목의 날: 갭 상승으로 시작하고 거래량이 평소의 몇 배로 터지는 저가(대략 $2~$20) 소형주. 그런 날이 아니면 패턴을 찾지 않습니다(요약의 장전 고가·전일 종가·거래량 배수로 판단).
- 깃대: 짧은 시간에 연속 양봉으로 가파르게 오른 구간(몇 % 이상).
- 깃발: 1~4봉의 얕은 눌림 — 거래량이 줄고, 9 EMA 위에서 버티고, 깃대의 절반 이상을 되돌리지 않습니다. 깊게 밀리면 깃발이 아닙니다.
- 진입: 눌림 뒤 처음으로 직전 봉의 고가 위로 마감하는 봉(첫 신고가 캔들). 장전 고가 위에서 나오면 더 좋습니다.
- 손절: 깃발 저점 바로 아래(닿으면 손절 — touch). 목표: 위험의 2배 이상, 보통 당일 고가 재시험. 손익비가 2:1 이 안 나오면 하지 않습니다.
- 대략 정오 전에만 진입하고, 장 마감 전에 정리합니다. 피할 것: 깃대 없이 지지부진한 상승, 대량 거래 음봉이 섞인 눌림, VWAP·9 EMA 아래로 무너진 뒤의 반등."""

_MATT_DIAMOND = """Matt Diamond 의 대형주 Bull Flag — 장 초반 추세일의 10 EMA 눌림 (1분봉).
- 종목의 날: 장전 고가 위, 그리고 일봉의 최근 저항(며칠 고점) 위에서 시작해 장 초반에 힘 있게 뻗는 날(오프닝 드라이브 깃대).
- 눌림: 1~3봉이 10 EMA 근처까지 조용히 내려옵니다. 장전 고가 위를 지켜야 합니다.
- 진입: Green-Take-Red — 직전 음봉의 몸통을 삼키며 올라가는 양봉이 거래량을 늘리며 나오면, 그 양봉의 고가를 넘는 순간 삽니다(그 고가 위 마감을 조건으로 적거나, 이미 넘었으면 market).
- 손절: 그 양봉의 저가 아래(변동성이 크면 조금 더 여유). 목표: 위험의 약 2.5배. 1R 수익이면 손절을 본전으로, 10봉쯤 지나도 +0.5R 을 못 가면 정리를 생각합니다.
- 하루에 눌림 진입은 두 번 정도까지. 대략 14시 30분 전에만 진입하고, 장 마감 전에 정리합니다.
- 피할 것: 10 EMA 를 깊게 깨는 눌림, 장전 고가 아래로 떨어진 뒤의 반등, 오프닝 드라이브가 없는 밋밋한 날."""


PATTERNS: dict[str, PatternSpec] = {p.key: p for p in [
    PatternSpec("vwap_sr", "VWAP 지지·저항 돌파 (15분봉)", "55 VWAP SR Breakout 15m", ChartSpec("15m", vwap=True), _VWAP_SR,
                source="av_ext", ticker="QQQ", days=5, warmup_days=6, step=4, max_hold=128, regular_only=False,
                note="시간외 포함 15분봉(04:00–20:00). 평일 밤은 들고 간다."),
    PatternSpec("wedge_pop", "Wedge Pop (일봉)", "9 Wedge Pop Strategy", ChartSpec("1d", vwap=False, emas=(10, 20, 50, 200), view_bars=160, table_bars=60), _WEDGE_POP,
                source="daily", ticker="AAPL", days=120, warmup_days=420, step=1, max_hold=60),
    PatternSpec("wedge_pop_15m", "Wedge Pop (15분봉)", "10 Wedge Pop Strategy 15m", ChartSpec("15m", vwap=False, emas=(26, 78), view_bars=156, table_bars=60), _WEDGE_POP
                + "\n- 15분봉 판: 빠른 EMA = 26봉(하루), 느린 EMA = 78봉(3일). 정규장 봉만 봅니다. 며칠 들고 갈 수 있습니다.",
                source="stack_rth", ticker="AAPL", days=10, warmup_days=10, step=4, max_hold=260, regular_only=True),
    PatternSpec("wick_play", "Wick Play (일봉)", "5 Wick Play Strategy", ChartSpec("1d", vwap=False, emas=(10, 20, 50), view_bars=120, table_bars=50), _WICK_PLAY,
                source="daily", ticker="AAPL", days=120, warmup_days=420, step=1, max_hold=40),
    PatternSpec("wick_play_15m", "Wick Play (15분봉)", "11 Wick Play Strategy 15m", ChartSpec("15m", vwap=False, emas=(10, 26), view_bars=130, table_bars=52), _WICK_PLAY
                + "\n- 15분봉 판: 추세 기준선은 26봉 EMA(하루). 정규장 봉만 봅니다. 3봉 패턴이라 자주 확인합니다.",
                source="stack_rth", ticker="AAPL", days=5, warmup_days=8, step=2, max_hold=260, regular_only=True),
    PatternSpec("fvg", "Fair Value Gap (15분봉)", "13 Fair Value Gap", ChartSpec("15m", vwap=True, view_bars=140, table_bars=56), _FVG_US,
                source="stack_all", ticker="AAPL", days=5, warmup_days=6, step=2, flat_at_close=True),
    PatternSpec("fvg_btc", "Bitcoin Fair Value Gap (15분봉)", "15 Bitcoin Fair Value Gap", ChartSpec("15m", vwap=False, session="crypto", view_bars=160, table_bars=64), _FVG_BTC,
                source="crypto", ticker="BTC-USD.CC", days=3, warmup_days=4, step=4, max_hold=192),
    PatternSpec("scraface", "Scraface ORB 재시험 (1분봉)", "16 Scraface Pattern", ChartSpec("1m", vwap=True, view_bars=150, table_bars=60), _SCRAFACE,
                source="stack_rth", ticker="TSLA", days=3, warmup_days=4, step=3, flat_at_close=True, regular_only=True, ask_from="09:36", ask_until="11:00"),
    PatternSpec("tradesharp_orb", "TradeSharp ORB (5분봉)", "18 TradeSharp ORB", ChartSpec("5m", vwap=True, view_bars=156, table_bars=60), _TRADESHARP,
                source="stack_rth", ticker="QQQ", days=5, warmup_days=6, step=2, flat_at_close=True, regular_only=True, ask_from="09:45", ask_until="11:30"),
    PatternSpec("first_candle", "First Candle Rule (1분봉)", "19 First Candle Rule", ChartSpec("1m", vwap=True, view_bars=150, table_bars=60), _FIRST_CANDLE,
                source="stack_rth", ticker="QQQ", days=3, warmup_days=4, step=3, flat_at_close=True, regular_only=True, ask_from="09:35", ask_until="11:00"),
    PatternSpec("bull_flag", "Bull Flag — Ross Cameron (1분봉)", "21 Bull Flag Strategy", ChartSpec("1m", vwap=True, emas=(9,), view_bars=150, table_bars=60), _BULL_FLAG,
                source="stack_all", ticker="OSR", days=3, warmup_days=4, step=3, flat_at_close=True, regular_only=True, ask_from="09:31", ask_until="12:00",
                note="유통 주식 수(float) 조건은 자료가 없어 모델에게 주지 않는다."),
    PatternSpec("matt_diamond", "Matt Diamond Bull Flag (1분봉)", "23 Matt Diamond Bull Flag", ChartSpec("1m", vwap=True, emas=(10,), view_bars=150, table_bars=60), _MATT_DIAMOND,
                source="stack_all", ticker="TSLA", days=3, warmup_days=4, step=3, flat_at_close=True, regular_only=True, ask_from="09:31", ask_until="14:30"),
]}
