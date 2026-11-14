"""SOXL 사이클 탐색 — "업계 사람이 읽는 숫자" 수집 (월간, 관측월 기준 wide parquet).

 한국은행 ECOS: 반도체 수출물가(달러, 1971~)·반도체 수출금액지수(2000~)·컴퓨터전자광학 수출금액지수(1988~)·전자부품 출하/재고 지수(계절조정, 1975~)
 FRED: 미국 반도체 산업생산·가동률·생산자물가, 컴퓨터전자 신규주문·출하, 한국 총수출, 장단기 금리차, 신용 스프레드
수정된 최종값을 받는다(당시 발표본 아님) → 쓰는 쪽에서 발표 지연을 넉넉히(관측월 말 + lag 개월) 둔다. 물가·지수류는 수정 폭이 작다.
실행: docker compose exec backtester python soxl_ind_cli.py '{}'  → /app/sweep_results/soxl_indicators.parquet
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

SR = Path("/app/sweep_results")
ECOS = {"kr_semi_px": ("402Y014", ["3091AA", "D"]), "kr_semi_exp": ("403Y001", ["3091AA"]), "kr_it_exp": ("403Y001", ["309AA"]),
        "kr_elec_ship": ("901Y032", ["I11ACQ", "4"]), "kr_elec_inv": ("901Y032", ["I11ACQ", "6"]), "kr_elec_prod": ("901Y032", ["I11ACQ", "2"])}
FRED = {"us_semi_ip": "IPG3344S", "us_semi_util": "CAPUTLG3344S", "us_semi_ppi": "PCU334413334413", "us_ce_orders": "A34SNO", "us_ce_ship": "A34SVS",
        "kr_exports": "XTEXVA01KRM667S", "t10y3m": "T10Y3M", "baa10y": "BAA10Y", "fedfunds": "FEDFUNDS"}


def ecos(code: str, items: list[str], key: str) -> pd.Series:
    out = {}
    for a in range(1, 2000, 1000):
        u = f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/{a}/{a + 999}/{code}/M/197001/202612/" + "/".join(items)
        rows = requests.get(u, timeout=120).json().get("StatisticSearch", {}).get("row", [])
        out.update({pd.Timestamp(r["TIME"][:4] + "-" + r["TIME"][4:] + "-01"): float(r["DATA_VALUE"]) for r in rows if r["DATA_VALUE"]})
        if len(rows) < 1000:
            break
    return pd.Series(out).sort_index()


def fred(sid: str, key: str) -> pd.Series:
    j = requests.get("https://api.stlouisfed.org/fred/series/observations", params={"series_id": sid, "api_key": key, "file_type": "json", "observation_start": "1970-01-01"}, timeout=120).json()
    s = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in j["observations"] if o["value"] not in (".", "")}).sort_index()
    return s.resample("MS").mean() if len(s) > 2000 else s          # 일간(금리차 등)은 월평균


def main(cfg):
    kb, kf = os.environ["BOK_API_KEY"], os.environ["FRED_API_KEY"]; cols = {}
    for name, (code, items) in ECOS.items():
        cols[name] = ecos(code, items, kb); print(name, len(cols[name]), cols[name].index[0].date(), cols[name].index[-1].date(), file=sys.stderr, flush=True)
    for name, sid in FRED.items():
        cols[name] = fred(sid, kf); print(name, len(cols[name]), cols[name].index[0].date(), cols[name].index[-1].date(), file=sys.stderr, flush=True)
    df = pd.DataFrame(cols).sort_index(); df.to_parquet(cfg.get("out", SR / "soxl_indicators.parquet")); print(df.tail(3).T.to_string())


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
