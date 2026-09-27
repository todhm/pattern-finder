"""리더 로테이션 유니버스(3,934종목) 조정종가/거래대금 벌크 다운로드.

alphafolio us_daily는 비조정 종가 + 장중 변화율뿐이라 총수익 복원이
불가 → yfinance auto_adjust 종가를 청크로 받아 parquet 매트릭스 생성.
청크별 partial 저장으로 중단 후 재실행 시 이어받는다.

실행: docker compose exec backtester python fetch_leader_universe.py
"""

import os
import sys
import time

import pandas as pd
import yfinance as yf

SYMBOLS_SRC = "sweep_results/leader_rotation_daily.csv.gz"
PART_DIR = "sweep_results/leader_parts"
OUT_CLOSE = "sweep_results/leader_rotation_close.parquet"
OUT_DVOL = "sweep_results/leader_rotation_dollarvol.parquet"
CHUNK = 150
START = "2017-11-01"


def main() -> None:
    symbols = sorted(
        pd.read_csv(SYMBOLS_SRC, usecols=["symbol"])["symbol"].unique()
    )
    os.makedirs(PART_DIR, exist_ok=True)
    chunks = [symbols[i : i + CHUNK] for i in range(0, len(symbols), CHUNK)]
    print(f"{len(symbols)} symbols, {len(chunks)} chunks", file=sys.stderr)

    for ci, chunk in enumerate(chunks):
        part = f"{PART_DIR}/part_{ci:03d}.parquet"
        if os.path.exists(part):
            continue
        for attempt in range(3):
            try:
                raw = yf.download(
                    " ".join(chunk), start=START, auto_adjust=True,
                    progress=False, threads=True, group_by="column",
                )
                break
            except Exception as e:  # noqa: BLE001
                print(f"chunk {ci} retry {attempt}: {e}", file=sys.stderr)
                time.sleep(15 * (attempt + 1))
        else:
            print(f"chunk {ci} FAILED — skip", file=sys.stderr)
            continue
        if raw is None or raw.empty:
            print(f"chunk {ci} empty", file=sys.stderr)
            continue
        close = raw["Close"].astype("float32")
        dvol = (raw["Close"] * raw["Volume"]).astype("float32")
        merged = pd.concat({"close": close, "dvol": dvol}, axis=1)
        merged.to_parquet(part)
        got = close.notna().any().sum()
        print(f"chunk {ci + 1}/{len(chunks)}: {got}/{len(chunk)} tickers",
              file=sys.stderr)
        time.sleep(1.0)

    closes, dvols = [], []
    for fn in sorted(os.listdir(PART_DIR)):
        m = pd.read_parquet(f"{PART_DIR}/{fn}")
        closes.append(m["close"])
        dvols.append(m["dvol"])
    close_all = pd.concat(closes, axis=1)
    close_all = close_all.loc[:, close_all.notna().any()]
    close_all = close_all.loc[:, ~close_all.columns.duplicated()]
    dvol_all = pd.concat(dvols, axis=1)
    dvol_all = dvol_all.loc[:, close_all.columns]
    dvol20 = dvol_all.rolling(20, min_periods=5).mean().astype("float32")
    close_all.sort_index().to_parquet(OUT_CLOSE)
    dvol20.sort_index().to_parquet(OUT_DVOL)
    print(f"saved {OUT_CLOSE} {close_all.shape}, {OUT_DVOL}", file=sys.stderr)


if __name__ == "__main__":
    main()
