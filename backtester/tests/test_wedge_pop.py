import pandas as pd

from pattern.services.wedge_pop import WedgePopDetector


class TestWedgePopDetector:
    def setup_method(self):
        self.detector = WedgePopDetector(
            consolidation_days=10,
            volume_multiplier=1.3,
            atr_compression=0.8,
        )

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

    def test_signal_metadata_has_required_keys(
        self, consolidation_with_breakout: pd.DataFrame
    ):
        signals = self.detector.detect(consolidation_with_breakout)
        if signals:
            meta = signals[0].metadata
            assert "atr_ratio" in meta
            assert "volume_ratio" in meta
            assert "consolidation_low" in meta

    def test_stricter_params_fewer_signals(
        self, consolidation_with_breakout: pd.DataFrame
    ):
        strict = WedgePopDetector(
            consolidation_days=15,
            volume_multiplier=3.0,
            atr_compression=0.5,
        )
        lenient = WedgePopDetector(
            consolidation_days=5,
            volume_multiplier=1.0,
            atr_compression=1.0,
        )
        assert len(strict.detect(consolidation_with_breakout)) <= len(
            lenient.detect(consolidation_with_breakout)
        )
