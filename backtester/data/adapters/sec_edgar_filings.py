"""SEC EDGAR 공시 원문 — 기준일 전에 제출된 최신 10-K(없으면 20-F)·위임장(DEF 14A)의 본문 발췌와 8-K 항목 목록.

피셔 15 포인트 문서 판독(`fisher_cli.py dossier`)용. 받은 원문은 `cache_dir` 에 저장한다(재실행 시 SEC 재요청 없음).
SEC 는 초당 10건 이하를 요구한다 — 요청 사이 0.15초 쉰다. User-Agent 에는 개인 정보를 넣지 않는다.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

UA = {"User-Agent": "research-script/1.0 (academic backtest)"}
_ITEMS = [("1", r"business"), ("1a", r"risk\s+factors"), ("1b", r"unresolved"), ("1c", r"cybersecurity"), ("2", r"properties"), ("3", r"legal\s+proceedings"),
          ("4", r"mine\s+safety"), ("5", r"market\s+for"), ("6", r"(?:\[?reserved\]?|selected)"), ("7", r"management.{0,3}s\s+discussion"), ("7a", r"quantitative\s+and\s+qualitative"),
          ("8", r"financial\s+statements"), ("9", r"changes\s+in\s+and\s+disagreements"), ("9a", r"controls\s+and\s+procedures"), ("9b", r"other\s+information"),
          ("10", r"directors")]
_SUFFIX = r"(?:,?\s+(?:inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|n\.?v\.?|s\.?a\.?|llc|lp|class\s+[a-c])\.?)+$"


class SecEdgarFilings:
    def __init__(self, cache_dir: str | Path, pause: float = 0.15):
        self.dir = Path(cache_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.pause = pause

    def _get(self, url: str, name: str, binary: bool = False):
        p = self.dir / name
        if p.exists():
            return p.read_bytes() if binary else p.read_text(encoding="utf-8", errors="replace")
        time.sleep(self.pause)
        r = requests.get(url, headers=UA, timeout=90)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        p.write_bytes(r.content)
        return r.content if binary else r.content.decode("utf-8", errors="replace")

    def filings(self, cik: int, since: str | None = None) -> tuple[dict, list[dict]]:
        """(회사 정보, 공시 목록[{form, filingDate, accession, doc, items}]) — 최신순.

        SEC 의 `recent` 는 최근 약 1,000건뿐이다(공시가 많은 회사는 몇 해 전 10-K 가 빠진다). `since`(YYYY-MM-DD)를 주면 그 날짜 이후를 덮는 옛 목록 파일도 읽는다.
        """
        txt = self._get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", f"sub_{int(cik)}.json")
        if not txt:
            return {}, []
        j = json.loads(txt)
        blocks = [j["filings"]["recent"]]
        if since:
            for x in j["filings"].get("files", []):
                if x.get("filingTo", "") >= since:
                    more = self._get(f"https://data.sec.gov/submissions/{x['name']}", f"sub_{x['name']}")
                    if more:
                        blocks.append(json.loads(more))
        rows = [{"form": f["form"][i], "filingDate": f["filingDate"][i], "accession": f["accessionNumber"][i], "doc": f["primaryDocument"][i], "items": f["items"][i]}
                for f in blocks for i in range(len(f["form"]))]
        rows.sort(key=lambda r: r["filingDate"], reverse=True)
        info = {"name": j.get("name", ""), "former": [x.get("name", "") for x in j.get("formerNames", [])], "sic": j.get("sic"), "sic_desc": j.get("sicDescription", ""),
                "tickers": j.get("tickers", []), "state": j.get("stateOfIncorporation", "")}
        return info, rows

    def document_text(self, cik: int, accession: str, doc: str) -> str:
        acc = accession.replace("-", "")
        name = f"doc_{int(cik)}_{acc}.txt"
        p = self.dir / name
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
        html = self._get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}", f"raw_{int(cik)}_{acc}.htm")
        if not html:
            return ""
        text = html_to_text(html)
        p.write_text(text, encoding="utf-8")
        (self.dir / f"raw_{int(cik)}_{acc}.htm").unlink(missing_ok=True)       # 원본 html 은 크니 글만 남긴다
        return text


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "ix:header", "head"]):
        t.decompose()
    for t in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        t.decompose()
    for td in soup.find_all(["td", "th"]):
        td.append(" | ")
    for tr in soup.find_all(["tr", "p", "div", "br", "li", "h1", "h2", "h3", "h4"]):
        tr.append("\n")
    text = soup.get_text()
    text = text.replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?:\s*\|\s*){2,}", " | ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def split_items(text: str) -> dict[str, str]:
    """10-K 본문을 Item 별로 자른다. 목차의 같은 제목은 구간이 짧으므로, 제목이 여러 번 나오면 **가장 긴 구간**을 본문으로 본다."""
    heads: list[tuple[int, str]] = []
    for key, pat in _ITEMS:
        for m in re.finditer(rf"(?im)^[\s|]*(?:part\s+[iv]+[\s|\.,:\-–—]*)?item[\s|]*{key}[\s|]*[\.:\-–—]?[\s|]*{pat}", text):
            heads.append((m.start(), key))
    heads.sort()
    out: dict[str, str] = {}
    for n, (pos, key) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(text)
        seg = text[pos:end]
        if len(seg) > len(out.get(key, "")):
            out[key] = seg
    return out


def window(text: str, pattern: str, size: int, before: int = 0) -> str:
    """pattern 이 처음 '제목처럼'(줄 머리) 나오는 곳부터 size 글자. 없으면 빈 글. 여러 번이면 뒤따르는 글이 가장 긴(다음 제목까지) 것 대신 가장 뒤의 것(목차 회피)."""
    ms = list(re.finditer(rf"(?im)^\s*{pattern}", text))
    if not ms:
        return ""
    m = ms[-1] if len(ms) > 1 else ms[0]
    return text[max(0, m.start() - before): m.start() + size]


def mask_company(text: str, names: list[str], tickers: list[str]) -> str:
    """회사 이름·티커를 가린다(완전하지 않다 — 제품명·지명은 남는다)."""
    out = text
    cores = set()
    for nm in names:
        nm = (nm or "").strip()
        if not nm:
            continue
        core = re.sub(_SUFFIX, "", nm, flags=re.I).strip(" ,.")
        for c in {nm, core}:
            if len(c) >= 3:
                cores.add(c)
    for c in sorted(cores, key=len, reverse=True):
        out = re.sub(re.escape(c).replace(r"\ ", r"\s+"), "[회사]", out, flags=re.I)
    for t in tickers:
        if len(t) >= 2:
            out = re.sub(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", "[티커]", out)
    return out
