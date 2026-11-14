"""급등주 탐색 S6 — SEC "Insider Transactions Data Sets"(Form 3/4/5 분기 묶음) 수집·정리.

Alpha Vantage 의 EARNINGS·INSIDER_TRANSACTIONS 는 폐지 종목에 빈 응답을 준다(IMMU·LVGO·MNTA·CPE 확인) → 생존 편향.
SEC 분기 묶음은 폐지 회사도 그대로 있고 거래 코드(P = 공개시장 매수, S = 매도)가 구분된다.

  fetch  분기 zip 을 sweep_results/sec_insider_cache/ 로 (개발 구간 2016q1~2023q4 만; 봉인 구간은 받지 않는다)
  build  Form 4 비파생 P·S 거래 → sweep_results/surge_insider_tx.parquet (공시 한 건·코드당 한 줄)

실행: docker compose exec backtester python surge_insider_cli.py fetch '{}'
      docker compose exec backtester python surge_insider_cli.py build '{}'
"""

from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

from strategy.adapters.insider_signals import normalize_symbol

SR = Path("/app/sweep_results")
URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
UA = {"User-Agent": "research-script/1.0 (academic backtest)"}


def quarters(first: str, last: str) -> list[str]:
    y0, q0 = int(first[:4]), int(first[5]); y1, q1 = int(last[:4]), int(last[5])
    return [f"{y}q{q}" for y in range(y0, y1 + 1) for q in range(1, 5) if (y, q) >= (y0, q0) and (y, q) <= (y1, q1)]


def cmd_fetch(cfg):
    import requests
    cache = Path(cfg.get("cache", SR / "sec_insider_cache")); cache.mkdir(parents=True, exist_ok=True)
    for q in quarters(cfg.get("first", "2016q1"), cfg.get("last", "2023q4")):
        fn = cache / f"{q}_form345.zip"
        if fn.exists():
            continue
        r = requests.get(URL.format(q=q), headers=UA, timeout=180)
        if r.status_code != 200:
            print(f"{q}: HTTP {r.status_code}", file=sys.stderr, flush=True); continue
        fn.write_bytes(r.content); print(f"{q}: {len(r.content) / 1e6:.1f} MB", file=sys.stderr, flush=True); time.sleep(0.5)
    print(f"{len(list(cache.glob('*.zip')))} 분기 · {cache}")


def _one(path: Path) -> pd.DataFrame:
    z = zipfile.ZipFile(path); rd = lambda n, cols: pd.read_csv(z.open(n), sep="\t", usecols=cols, low_memory=False)  # noqa: E731
    sub = rd("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"])
    sub = sub[sub["DOCUMENT_TYPE"].astype(str) == "4"]
    own = rd("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
    own = own.groupby("ACCESSION_NUMBER").agg(owner=("RPTOWNERCIK", "first"), rel=("RPTOWNER_RELATIONSHIP", lambda s: ",".join(sorted(set(map(str, s))))),
                                              title=("RPTOWNER_TITLE", lambda s: " / ".join(str(x) for x in s if isinstance(x, str))))
    tr = rd("NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"])
    tr = tr[tr["TRANS_CODE"].isin(["P", "S"])].copy()
    tr["value"] = tr["TRANS_SHARES"].astype(float) * tr["TRANS_PRICEPERSHARE"].astype(float)
    tr["TRANS_DATE"] = pd.to_datetime(tr["TRANS_DATE"], format="%d-%b-%Y", errors="coerce")
    tr = tr.groupby(["ACCESSION_NUMBER", "TRANS_CODE"]).agg(trans_date=("TRANS_DATE", "max"), shares=("TRANS_SHARES", "sum"), value=("value", "sum")).reset_index()
    d = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, left_on="ACCESSION_NUMBER", right_index=True, how="left")
    d["filing_date"] = pd.to_datetime(d["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    d["symbol"] = d["ISSUERTRADINGSYMBOL"].map(normalize_symbol)
    rel = d["rel"].fillna(""); title = d["title"].fillna("").str.upper()
    d["od"] = rel.str.contains("Officer|Director", case=False)
    d["ceo"] = title.str.contains(r"\bCEO\b|CHIEF EXECUTIVE|\bCFO\b|CHIEF FINANCIAL|PRESIDENT")
    d = d.rename(columns={"ACCESSION_NUMBER": "accession", "TRANS_CODE": "code", "ISSUERCIK": "issuer_cik"})
    return d.dropna(subset=["symbol", "filing_date"])[["accession", "symbol", "issuer_cik", "filing_date", "trans_date", "owner", "rel", "title", "od", "ceo", "code", "shares", "value"]]


def cmd_build(cfg):
    cache = Path(cfg.get("cache", SR / "sec_insider_cache")); out = Path(cfg.get("out", SR / "surge_insider_tx.parquet"))
    parts = []
    for fn in sorted(cache.glob("*_form345.zip")):
        parts.append(_one(fn)); print(fn.name, len(parts[-1]), file=sys.stderr, flush=True)
    d = pd.concat(parts, ignore_index=True).drop_duplicates(["accession", "code"])
    d = d[(d["value"] > 0) & (d["value"] < 5e9)]
    d.to_parquet(out)
    b = d[d["code"] == "P"]
    print(f"{len(d):,} 줄 (매수 {len(b):,} · 임원·이사 매수 {int(b['od'].sum()):,}) · 종목 {d['symbol'].nunique():,} · {d['filing_date'].min().date()}~{d['filing_date'].max().date()} → {out}")


if __name__ == "__main__":
    {"fetch": cmd_fetch, "build": cmd_build}[sys.argv[1]](json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
