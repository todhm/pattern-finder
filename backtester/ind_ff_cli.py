"""산업군 성장 탐색 G0 — Kenneth French 자료실의 49 산업 포트폴리오(1926-07~, 월간)와 시장·무위험 수익을 받아 parquet 로.

산업 포트폴리오는 그 달 그 산업(SIC)에 있던 모든 상장사로 만들어져 폐지 종목이 들어 있다(생존 편향 없음).
 ind_ff49.parquet  : date(월말), ind, vw(시총가중 월수익), ew(동일가중), n(종목 수), size(평균 시총 $M), bm(연 1회 BE/ME, 7월~다음 6월에 적용)
 ind_ff_mkt.parquet: date, mkt(시장 수익 = Mkt-RF + RF), rf
실행: docker compose exec backtester python ind_ff_cli.py '{}'
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SR = Path("/app/sweep_results")
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
UA = {"User-Agent": "research-script/1.0 (academic backtest)"}


def _text(name: str) -> str:
    r = requests.get(BASE + name, headers=UA, timeout=120); r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content)); return z.read(z.namelist()[0]).decode("latin-1")


def sections(txt: str) -> dict[str, pd.DataFrame]:
    """French CSV 는 빈 줄로 나뉜 여러 표. 표 제목 → DataFrame(index = YYYYMM 또는 YYYY 정수)."""
    out, title, rows, head = {}, "", [], None
    for line in txt.splitlines() + [""]:
        s = line.strip()
        if not s:
            if head is not None and rows:
                out[title] = pd.DataFrame([r[1:] for r in rows], index=[int(r[0]) for r in rows], columns=head).astype(float)
            rows, head = [], None; continue
        parts = [p.strip() for p in s.split(",")]
        if head is None:
            if parts[0] == "" and len(parts) > 2:
                head = parts[1:]
            else:
                title = s
        elif parts[0].isdigit():
            rows.append(parts)
    return out


def main(cfg):
    sec = sections(_text("49_Industry_Portfolios_CSV.zip")); keys = list(sec); print(keys, file=sys.stderr)
    pick = lambda *w: sec[next(k for k in keys if all(x in k for x in w))]  # noqa: E731
    vw, ew = pick("Value Weighted", "Monthly"), pick("Equal Weighted", "Monthly"); n, size, bm = pick("Number of Firms"), pick("Average Firm Size"), pick("BE")
    idx = pd.to_datetime(vw.index.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    clean = lambda d: d.where(d > -99).set_axis(idx[: len(d)]) if len(d) == len(idx) else d  # noqa: E731
    vw, ew, n, size = clean(vw) / 100.0, clean(ew) / 100.0, clean(n), clean(size); n = n.where(n > 0); bm = bm.where(bm > -99)
    # BE/ME: y 년 값(회계연도 y-1 장부 ÷ y-1 년 12월 시총)은 y 년 7월~y+1 년 6월 포트폴리오에 쓰인다 → 6월 말부터 알 수 있다
    bmm = pd.DataFrame(index=idx, columns=vw.columns, dtype=float)
    for y, row in bm.iterrows():
        bmm.loc[(idx >= f"{y}-06-30") & (idx < f"{y + 1}-06-30")] = row.to_numpy()
    long = pd.concat({"vw": vw.stack(), "ew": ew.stack(), "n": n.stack(), "size": size.stack(), "bm": bmm.stack()}, axis=1).reset_index(names=["date", "ind"])
    long.to_parquet(SR / "ind_ff49.parquet")
    f = sections(_text("F-F_Research_Data_Factors_CSV.zip")); fm = next(v for k, v in f.items() if len(v) > 600)
    fi = pd.to_datetime(fm.index.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    mk = pd.DataFrame({"mkt": (fm["Mkt-RF"] + fm["RF"]).to_numpy() / 100.0, "rf": fm["RF"].to_numpy() / 100.0}, index=fi); mk.to_parquet(SR / "ind_ff_mkt.parquet")
    print(long.groupby("ind")["vw"].count().describe(), long["date"].min(), long["date"].max(), mk.index.min(), mk.index.max(), np.round((1 + mk["mkt"]).prod(), 0))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
