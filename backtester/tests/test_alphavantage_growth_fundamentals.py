"""AlphaVantageGrowthFundamentalsAdapter — 분기·연간 파싱 + 캐시 버전 테스트 (네트워크 없음)."""

from __future__ import annotations

from datetime import date

from data.adapters.alphavantage_growth_fundamentals import (
    CACHE_VERSION,
    AlphaVantageGrowthFundamentalsAdapter,
)


def _adapter(tmp_path) -> AlphaVantageGrowthFundamentalsAdapter:
    return AlphaVantageGrowthFundamentalsAdapter(
        api_key="test", cache_dir=tmp_path, include_profile=False,
    )


_RAW = {
    "quarterlyEarnings": [
        {"fiscalDateEnding": "2019-06-30", "reportedDate": "2019-08-08",
         "reportedEPS": "0.34", "estimatedEPS": "0.25"},
        {"fiscalDateEnding": "2019-03-31", "reportedDate": "2019-05-02",
         "reportedEPS": "0.60", "estimatedEPS": "0.57"},
    ],
    "quarterlyReports": [
        {"fiscalDateEnding": "2019-06-30", "totalRevenue": "120373000", "netIncome": "10241000"},
    ],
    "annualEarnings": [
        {"fiscalDateEnding": "2019-06-30", "reportedEPS": "1.37"},
        {"fiscalDateEnding": "2018-06-30", "reportedEPS": "1.30"},
        {"fiscalDateEnding": "2017-06-30", "reportedEPS": "None"},
    ],
    "annualReports": [
        {"fiscalDateEnding": "2019-06-30", "totalRevenue": "467633000", "netIncome": "53823000"},
        {"fiscalDateEnding": "2018-06-30", "totalRevenue": "377527000", "netIncome": "38598000"},
    ],
    "Sector": "TECHNOLOGY", "Industry": "SOFTWARE - APPLICATION",
}


def test_parse_annuals_with_q4_report_date(tmp_path):
    snap = _adapter(tmp_path)._parse("PCTY", _RAW)
    assert [a.fiscal_date for a in snap.annuals] == [
        date(2017, 6, 30), date(2018, 6, 30), date(2019, 6, 30)]
    fy19 = snap.annuals[-1]
    assert fy19.eps == 1.37 and fy19.revenue == 467633000.0 and fy19.net_income == 53823000.0
    # 연간 발표일 = 같은 회계연도 말 Q4 분기의 reportedDate
    assert fy19.report_date == date(2019, 8, 8)
    assert fy19.effective_report_date == date(2019, 8, 8)
    # Q4 분기 데이터가 없는 해는 발표일 None → +90일 보수 가정
    fy18 = snap.annuals[1]
    assert fy18.report_date is None
    assert fy18.effective_report_date == date(2018, 9, 28)
    # "None" 문자열 EPS → None, 매출 없음 → None
    fy17 = snap.annuals[0]
    assert fy17.eps is None and fy17.revenue is None and fy17.net_margin is None
    assert fy19.net_margin == 53823000.0 / 467633000.0
    # 분기 파싱은 그대로
    assert len(snap.quarters) == 2 and snap.quarters[-1].eps_estimate == 0.25
    assert snap.industry == "SOFTWARE - APPLICATION"


def test_cache_path_is_versioned_so_old_quarter_only_cache_is_ignored(tmp_path):
    a = _adapter(tmp_path)
    assert a._cache_path("PCTY").name == f"PCTY.{CACHE_VERSION}.json"
    assert CACHE_VERSION == "v2"


def test_fetch_uses_cached_slim_with_annuals(tmp_path):
    import json
    a = _adapter(tmp_path)
    a._cache_path("PCTY").write_text(json.dumps(_RAW))
    snap = a.fetch("PCTY")            # 네트워크 없이 캐시에서
    assert len(snap.annuals) == 3 and snap.annuals[-1].eps == 1.37


def test_fetch_balance_sheet_from_cache_with_report_dates(tmp_path):
    import json
    a = _adapter(tmp_path)
    a._cache_path("PCTY").write_text(json.dumps(_RAW))
    (tmp_path / "PCTY.bs.v1.json").write_text(json.dumps({"quarterlyReports": [
        {"fiscalDateEnding": "2019-06-30", "inventory": "None", "currentNetReceivables": "4358000"},
        {"fiscalDateEnding": "2019-03-31", "inventory": "0", "currentNetReceivables": "3000000"},
    ]}))
    bs = a.fetch_balance_sheet("PCTY")
    assert [b.fiscal_date for b in bs] == [date(2019, 3, 31), date(2019, 6, 30)]
    assert bs[-1].receivables == 4358000.0 and bs[-1].inventory is None
    assert bs[-1].report_date == date(2019, 8, 8)          # EARNINGS reportedDate로 채움
    assert bs[0].report_date == date(2019, 5, 2)
    assert bs[0].inventory == 0.0
