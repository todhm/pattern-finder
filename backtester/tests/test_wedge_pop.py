from datetime import date

import pandas as pd
import pytest

from pattern.adapters.wedge_pop import WedgePopDetector


def _make_df(rows: list[tuple]) -> pd.DataFrame:
    """Build DataFrame from (date_str, open, high, low, close, volume) rows."""
    dates = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {
            "Open": [r[1] for r in rows],
            "High": [r[2] for r in rows],
            "Low": [r[3] for r in rows],
            "Close": [r[4] for r in rows],
            "Volume": [r[5] for r in rows],
        },
        index=dates,
    )


# Real ELF data: Oct 20 – Nov 17, 2023
# Wedge pop breakout on 2023-11-14
ELF_DATA = _make_df([
    ("2023-10-20", 105.04, 105.04, 101.11, 101.88, 1231800),
    ("2023-10-23", 103.58, 107.91, 101.21, 105.52, 1362500),
    ("2023-10-24", 106.91, 112.47, 106.91, 110.23, 1139900),
    ("2023-10-25", 109.90, 110.69, 107.07, 108.44, 808700),
    ("2023-10-26", 109.39, 109.60, 102.31, 102.96, 1031400),
    ("2023-10-27", 102.95, 104.00, 101.01, 102.24, 958900),
    ("2023-10-30", 103.04, 104.40, 101.01, 103.15, 836000),
    ("2023-10-31", 101.63, 101.63, 88.47, 92.63, 4253500),
    ("2023-11-01", 91.48, 94.55, 89.50, 94.54, 2703300),
    ("2023-11-02", 105.58, 109.00, 90.34, 98.01, 5094100),
    ("2023-11-03", 99.55, 100.45, 93.12, 96.15, 2809900),
    ("2023-11-06", 97.66, 102.75, 95.52, 101.68, 1985700),
    ("2023-11-07", 101.39, 103.24, 98.20, 102.01, 1548200),
    ("2023-11-08", 101.56, 102.79, 97.63, 98.14, 1720500),
    ("2023-11-09", 97.62, 101.99, 95.77, 99.45, 1726200),
    ("2023-11-10", 98.98, 99.49, 92.16, 95.09, 2631400),
    ("2023-11-13", 94.98, 97.75, 93.55, 97.28, 1140200),
    ("2023-11-14", 101.00, 107.53, 100.04, 106.21, 1840200),
    ("2023-11-15", 106.80, 114.42, 106.76, 113.21, 2272500),
    ("2023-11-16", 110.16, 113.63, 108.94, 111.88, 1306300),
    ("2023-11-17", 112.75, 117.13, 111.00, 112.85, 2331100),
])

# Real AMZN data: Apr 15 – Jun 14, 2021
# Wedge pop breakout on 2021-06-08 (daily move +2.07%, EMA distance only +0.8%)
# Needs history from Apr for EMA convergence (peak ~173 → decline → consolidation ~160)
AMZN_2021_DATA = _make_df([
    ("2021-04-15", 168.55, 169.85, 167.60, 168.95, 64672000),
    ("2021-04-16", 169.00, 170.34, 167.78, 169.97, 63720000),
    ("2021-04-19", 169.52, 171.80, 168.01, 168.60, 54508000),
    ("2021-04-20", 168.68, 169.15, 165.80, 166.73, 52460000),
    ("2021-04-21", 165.80, 168.14, 165.19, 168.10, 44224000),
    ("2021-04-22", 168.58, 168.64, 165.07, 165.45, 51612000),
    ("2021-04-23", 165.96, 168.75, 165.43, 167.04, 63856000),
    ("2021-04-26", 167.40, 171.42, 166.55, 170.45, 97614000),
    ("2021-04-27", 172.17, 173.00, 169.90, 170.87, 76542000),
    ("2021-04-28", 171.74, 174.49, 171.25, 172.93, 92638000),
    ("2021-04-29", 175.26, 175.72, 171.75, 173.57, 153648000),
    ("2021-04-30", 176.26, 177.70, 173.12, 173.37, 140186000),
    ("2021-05-03", 174.24, 174.33, 168.63, 169.32, 117510000),
    ("2021-05-04", 167.81, 168.40, 163.61, 165.59, 108788000),
    ("2021-05-05", 166.94, 167.74, 163.22, 163.53, 74226000),
    ("2021-05-06", 163.50, 165.72, 162.36, 165.32, 88954000),
    ("2021-05-07", 165.95, 166.54, 164.45, 164.58, 94206000),
    ("2021-05-10", 164.12, 164.15, 159.50, 159.52, 116772000),
    ("2021-05-11", 156.81, 161.90, 156.37, 161.20, 92396000),
    ("2021-05-12", 159.25, 160.40, 156.65, 157.60, 98728000),
    ("2021-05-13", 159.27, 160.19, 156.65, 158.07, 67018000),
    ("2021-05-14", 159.28, 161.44, 159.15, 161.15, 66500000),
    ("2021-05-17", 162.30, 164.64, 161.73, 163.52, 74478000),
    ("2021-05-18", 164.63, 165.60, 161.52, 161.61, 56568000),
    ("2021-05-19", 159.75, 161.74, 159.20, 161.59, 53594000),
    ("2021-05-20", 162.22, 162.98, 161.81, 162.38, 52664000),
    ("2021-05-21", 162.50, 162.83, 159.85, 160.15, 82098000),
    ("2021-05-24", 160.77, 162.90, 160.52, 162.25, 48456000),
    ("2021-05-25", 163.33, 163.99, 160.69, 162.95, 65222000),
    ("2021-05-26", 163.73, 164.79, 162.93, 163.26, 47680000),
    ("2021-05-27", 162.80, 163.02, 161.50, 161.51, 51224000),
    ("2021-05-28", 162.10, 162.40, 160.99, 161.15, 46596000),
    ("2021-06-01", 162.18, 162.55, 160.45, 160.93, 48600000),
    ("2021-06-02", 161.15, 161.75, 160.40, 161.70, 40290000),
    ("2021-06-03", 160.21, 160.72, 159.20, 159.35, 47966000),
    ("2021-06-04", 160.60, 161.05, 159.94, 160.31, 44994000),
    ("2021-06-07", 159.87, 160.40, 158.61, 159.90, 44316000),
    ("2021-06-08", 161.13, 163.98, 160.90, 163.21, 68334000),
    ("2021-06-09", 163.64, 164.88, 163.54, 164.06, 49110000),
    ("2021-06-10", 164.10, 167.55, 164.06, 167.48, 69530000),
    ("2021-06-11", 167.48, 168.33, 166.67, 167.34, 56348000),
    ("2021-06-14", 167.34, 169.25, 166.77, 169.19, 51394000),
])


class TestWedgePopDetector:
    def setup_method(self):
        self.detector = WedgePopDetector()

    # --- real data tests ---

    def test_detects_elf_wedge_pop(self):
        """ELF 2023-11-14: textbook wedge pop after consolidation below EMAs."""
        signals = self.detector.detect(ELF_DATA)
        assert len(signals) > 0

        # The first signal should be on or around 2023-11-14
        first = signals[0]
        assert first.pattern_name == "wedge_pop"
        assert first.date == date(2023, 11, 14)
        assert first.stop_loss < first.entry_price

    def test_detects_amzn_2021_wedge_pop(self):
        """AMZN 2021-06-08: breakout via daily momentum (EMA distance small)."""
        signals = self.detector.detect(AMZN_2021_DATA)
        assert len(signals) > 0
        first = signals[0]
        assert first.date == date(2021, 6, 8)
        assert first.stop_loss < first.entry_price

    def test_elf_signal_metadata(self):
        signals = self.detector.detect(ELF_DATA)
        assert len(signals) > 0
        meta = signals[0].metadata
        assert "breakout_strength" in meta
        assert "volume_ratio" in meta
        assert "consolidation_low" in meta
        assert meta["breakout_strength"] > 0

    # --- synthetic data tests ---

    def test_detects_breakout_after_consolidation(
        self, consolidation_with_breakout: pd.DataFrame
    ):
        signals = self.detector.detect(consolidation_with_breakout)
        assert len(signals) > 0
        for s in signals:
            assert s.pattern_name == "wedge_pop"
            assert s.stop_loss < s.entry_price

    def test_no_signal_in_flat_market(self, flat_market: pd.DataFrame):
        signals = self.detector.detect(flat_market)
        assert len(signals) == 0

    def test_stricter_params_fewer_signals(
        self, consolidation_with_breakout: pd.DataFrame
    ):
        strict = WedgePopDetector(consolidation_pct=0.9, breakout_pct=0.10)
        lenient = WedgePopDetector(consolidation_pct=0.3, breakout_pct=0.005)
        assert len(strict.detect(consolidation_with_breakout)) <= len(
            lenient.detect(consolidation_with_breakout)
        )
