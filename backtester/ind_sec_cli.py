"""산업군 성장 탐색 G4 자료 — SEC Financial Statement Data Sets(2009Q1~, 분기 zip)에서 매출·이익·R&D·설비투자를 '제출된 그대로' 뽑는다.

제출일(filed)이 있어 그 시점에 실제로 공개돼 있던 숫자만 쓸 수 있고(수정 재작성 없음), 그때 제출하던 회사가 전부 들어 있어 폐지 편향이 없다.
 fetch : 분기 zip → sweep_results/sec_fs_cache/YYYYqN.parquet (cik, sic, form, filed, ddate, qtrs, tag, value)  — 10-K/10-Q, 연결 전체(coreg·segments 없음), USD
 sic49 : French 49 산업 SIC 정의 → sweep_results/ind_sic49.parquet (lo, hi, ind)
실행: docker compose exec backtester python ind_sec_cli.py fetch '{"from": "2009q1", "to": "2026q3"}'
"""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

SR = Path("/app/sweep_results"); CACHE = SR / "sec_fs_cache"
UA = {"User-Agent": "research-script/1.0 (academic backtest)"}
TAGS = {"Revenues": "rev", "SalesRevenueNet": "rev", "RevenueFromContractWithCustomerExcludingAssessedTax": "rev", "RevenueFromContractWithCustomerIncludingAssessedTax": "rev", "SalesRevenueGoodsNet": "rev", "SalesRevenueServicesNet": "rev",
        "NetIncomeLoss": "ni", "OperatingIncomeLoss": "op", "ResearchAndDevelopmentExpense": "rnd", "PaymentsToAcquirePropertyPlantAndEquipment": "capex", "Assets": "assets", "StockholdersEquity": "eq"}


TAGS2 = {**TAGS, "GrossProfit": "gp", "CostOfRevenue": "cogs", "CostOfGoodsAndServicesSold": "cogs", "CostOfGoodsSold": "cogs", "InventoryNet": "inv", "ContractWithCustomerLiabilityCurrent": "defrev", "DeferredRevenueCurrent": "defrev",
         "RevenueRemainingPerformanceObligation": "rpo", "EntityCommonStockSharesOutstanding": "shares", "WeightedAverageNumberOfDilutedSharesOutstanding": "dsh", "ShareBasedCompensation": "sbc", "NetCashProvidedByUsedInOperatingActivities": "cfo",
         "CashAndCashEquivalentsAtCarryingValue": "cash", "LongTermDebt": "debt", "LongTermDebtNoncurrent": "debt", "ProceedsFromIssuanceOfCommonStock": "issue", "AccountsReceivableNetCurrent": "ar", "Goodwill": "gw", "PaymentsForRepurchaseOfCommonStock": "buyback"}


def fetch2(cfg):
    """판단 자료(도시에)용 넓은 판: 항목 TAGS2, 회사 이름·제출 번호(adsh) 포함, 단위 USD·shares → sweep_results/sec_fs_cache2/"""
    cache = SR / "sec_fs_cache2"; cache.mkdir(exist_ok=True); qs = [f"{y}q{q}" for y in range(2009, 2027) for q in range(1, 5)]; qs = [q for q in qs if cfg.get("from", "2009q1") <= q <= cfg.get("to", "2026q3")]
    for q in qs:
        fn = cache / f"{q}.parquet"
        if fn.exists():
            continue
        r = requests.get(f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip", headers=UA, timeout=600)
        if r.status_code != 200:
            print(q, "HTTP", r.status_code, file=sys.stderr, flush=True); continue
        z = zipfile.ZipFile(io.BytesIO(r.content)); sub = pd.read_csv(z.open("sub.txt"), sep="\t", dtype=str, usecols=["adsh", "cik", "name", "sic", "form", "filed", "fp"], quoting=3)
        sub = sub[sub["form"].isin(["10-K", "10-Q", "20-F", "40-F"])]; parts = []; head = pd.read_csv(z.open("num.txt"), sep="\t", nrows=0).columns; use = [c for c in ("adsh", "tag", "coreg", "ddate", "qtrs", "uom", "value", "segments") if c in head]
        for ch in pd.read_csv(z.open("num.txt"), sep="\t", dtype=str, usecols=use, chunksize=2_000_000, quoting=3, on_bad_lines="skip"):
            ch = ch[ch["tag"].isin(TAGS2) & ch["coreg"].isna() & ch["uom"].isin(["USD", "shares"]) & ch["adsh"].isin(sub["adsh"])]
            if "segments" in ch:
                ch = ch[ch["segments"].isna()]
            parts.append(ch[["adsh", "tag", "ddate", "qtrs", "value"]])
        d = pd.concat(parts).merge(sub, on="adsh"); d["value"] = pd.to_numeric(d["value"], errors="coerce"); d["item"] = d["tag"].map(TAGS2)
        for c in ("filed", "ddate"):
            d[c] = pd.to_datetime(d[c], format="%Y%m%d", errors="coerce")
        d["qtrs"] = pd.to_numeric(d["qtrs"], errors="coerce"); d["cik"] = pd.to_numeric(d["cik"]); d["sic"] = pd.to_numeric(d["sic"], errors="coerce")
        d = d.dropna(subset=["value", "filed", "ddate"]).groupby(["cik", "name", "sic", "form", "filed", "adsh", "item", "ddate", "qtrs"], as_index=False)["value"].max()
        d.to_parquet(fn); print(q, len(d), d["cik"].nunique(), file=sys.stderr, flush=True)


def fetch(cfg):
    CACHE.mkdir(exist_ok=True); qs = [f"{y}q{q}" for y in range(2009, 2027) for q in range(1, 5)]; qs = [q for q in qs if cfg.get("from", "2009q1") <= q <= cfg.get("to", "2026q3")]
    for q in qs:
        fn = CACHE / f"{q}.parquet"
        if fn.exists():
            continue
        r = requests.get(f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip", headers=UA, timeout=600)
        if r.status_code != 200:
            print(q, "HTTP", r.status_code, file=sys.stderr, flush=True); continue
        z = zipfile.ZipFile(io.BytesIO(r.content)); sub = pd.read_csv(z.open("sub.txt"), sep="\t", dtype=str, usecols=["adsh", "cik", "sic", "form", "filed", "fp"])
        sub = sub[sub["form"].isin(["10-K", "10-Q", "10-K405", "10-KSB", "20-F", "40-F"])]; parts = []
        head = pd.read_csv(z.open("num.txt"), sep="\t", nrows=0).columns; use = [c for c in ("adsh", "tag", "coreg", "ddate", "qtrs", "uom", "value", "segments") if c in head]
        for ch in pd.read_csv(z.open("num.txt"), sep="\t", dtype=str, usecols=use, chunksize=2_000_000, quoting=3, on_bad_lines="skip"):
            ch = ch[ch["tag"].isin(TAGS) & ch["coreg"].isna() & (ch["uom"] == "USD") & ch["adsh"].isin(sub["adsh"])]
            if "segments" in ch:
                ch = ch[ch["segments"].isna()]
            parts.append(ch[["adsh", "tag", "ddate", "qtrs", "value"]])
        d = pd.concat(parts).merge(sub, on="adsh"); d["value"] = pd.to_numeric(d["value"], errors="coerce"); d["item"] = d["tag"].map(TAGS)
        for c in ("filed", "ddate"):
            d[c] = pd.to_datetime(d[c], format="%Y%m%d", errors="coerce")
        d["qtrs"] = pd.to_numeric(d["qtrs"], errors="coerce"); d["cik"] = pd.to_numeric(d["cik"]); d["sic"] = pd.to_numeric(d["sic"], errors="coerce")
        d.dropna(subset=["value", "filed", "ddate"]).drop(columns=["adsh"]).to_parquet(fn); print(q, len(d), d["cik"].nunique(), f"{len(r.content) / 1e6:.0f}MB", file=sys.stderr, flush=True)


def sic49(cfg):
    r = requests.get("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes49.zip", headers=UA, timeout=120); txt = zipfile.ZipFile(io.BytesIO(r.content)); txt = txt.read(txt.namelist()[0]).decode("latin-1")
    rows, ind = [], None
    for line in txt.splitlines():
        m = re.match(r"^\s*(\d+)\s+(\S+)\s+", line)
        if m:
            ind = m.group(2); continue
        m = re.match(r"^\s+(\d{4})-(\d{4})", line)
        if m and ind:
            rows.append((int(m.group(1)), int(m.group(2)), ind))
    d = pd.DataFrame(rows, columns=["lo", "hi", "ind"]); d.to_parquet(SR / "ind_sic49.parquet"); print(d["ind"].nunique(), len(d), d.head())


if __name__ == "__main__":
    {"fetch": fetch, "fetch2": fetch2, "sic49": sic49}[sys.argv[1]](json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
