import pandas as pd

from pattern.adapters.reversal_extension import ReversalExtensionDetector


class TestReversalExtensionDetector:
    def setup_method(self):
        self.detector = ReversalExtensionDetector(
            extension_pct=0.03,
            volume_multiplier=1.5,
            lookback=5,
        )

    def test_detects_reversal_in_downtrend(self, downtrend_with_reversal: pd.DataFrame):
        signals = self.detector.detect(downtrend_with_reversal)
        assert len(signals) > 0
        for s in signals:
            assert s.pattern_name == "reversal_extension"
            assert s.stop_loss < s.entry_price
            assert s.confidence > 0

    def test_no_signal_in_flat_market(self, flat_market: pd.DataFrame):
        signals = self.detector.detect(flat_market)
        assert len(signals) == 0

    def test_signal_metadata_has_required_keys(self, downtrend_with_reversal: pd.DataFrame):
        signals = self.detector.detect(downtrend_with_reversal)
        if signals:
            meta = signals[0].metadata
            assert "extension_pct" in meta
            assert "volume_ratio" in meta

    def test_custom_params(self, downtrend_with_reversal: pd.DataFrame):
        strict = ReversalExtensionDetector(
            extension_pct=0.15,
            volume_multiplier=3.0,
            lookback=10,
        )
        signals = strict.detect(downtrend_with_reversal)
        lenient = ReversalExtensionDetector(
            extension_pct=0.01,
            volume_multiplier=1.0,
            lookback=3,
        )
        lenient_signals = lenient.detect(downtrend_with_reversal)
        assert len(signals) <= len(lenient_signals)
