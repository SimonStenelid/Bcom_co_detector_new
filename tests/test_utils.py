"""Tests for utility modules."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from botminer.utils.time import (
    parse_timestamp, compute_hourly_histogram, compute_hourly_entropy,
    compute_gini_coefficient, compute_time_gaps, compute_gap_stats,
    detect_ladder_sweep, detect_schedule_regularity
)
from botminer.utils.signatures import (
    compute_payload_signature, compute_signature_stats,
    compute_route_diversity, detect_pattern_anomalies
)
from botminer.utils.stats import (
    compute_robust_stats, compute_rate_stats, compute_http_version_stats,
    compute_status_code_stats, compute_error_bursts, compute_hygiene_stats
)


class TestTimeUtils:
    """Test time utility functions."""
    
    def test_parse_timestamp_iso(self):
        """Test parsing ISO timestamp."""
        ts = parse_timestamp("2025-01-15T10:30:00Z")
        assert ts.tz is not None
        assert ts.hour == 10
        assert ts.minute == 30
    
    def test_parse_timestamp_simple(self):
        """Test parsing simple timestamp."""
        ts = parse_timestamp("2025-01-15 10:30:00")
        assert ts.hour == 10
        assert ts.minute == 30
    
    def test_compute_hourly_histogram(self):
        """Test hourly histogram computation."""
        hours = pd.Series([0, 0, 1, 1, 1, 2, 2, 2, 2])
        hist = compute_hourly_histogram(hours)
        
        assert len(hist) == 24
        assert hist[0] == 2/9  # 2 out of 9 events
        assert hist[1] == 3/9  # 3 out of 9 events
        assert hist[2] == 4/9  # 4 out of 9 events
        assert np.sum(hist) == pytest.approx(1.0, abs=1e-10)
    
    def test_compute_hourly_entropy(self):
        """Test hourly entropy computation."""
        # Uniform distribution should have high entropy
        uniform_hist = np.ones(24) / 24
        entropy_uniform = compute_hourly_entropy(uniform_hist)
        assert entropy_uniform > 4.0  # log2(24) ≈ 4.58
        
        # Concentrated distribution should have low entropy
        concentrated_hist = np.zeros(24)
        concentrated_hist[0] = 1.0
        entropy_concentrated = compute_hourly_entropy(concentrated_hist)
        assert entropy_concentrated == 0.0
    
    def test_compute_gini_coefficient(self):
        """Test Gini coefficient computation."""
        # Perfect equality
        equal_values = np.array([1, 1, 1, 1])
        gini_equal = compute_gini_coefficient(equal_values)
        assert gini_equal == 0.0
        
        # Perfect inequality
        unequal_values = np.array([0, 0, 0, 10])
        gini_unequal = compute_gini_coefficient(unequal_values)
        assert gini_unequal > 0.5
    
    def test_compute_time_gaps(self):
        """Test time gap computation."""
        timestamps = pd.Series([
            datetime(2025, 1, 15, 10, 0, 0),
            datetime(2025, 1, 15, 10, 0, 5),
            datetime(2025, 1, 15, 10, 0, 15),
            datetime(2025, 1, 15, 10, 0, 20)
        ])
        
        gaps = compute_time_gaps(timestamps)
        expected_gaps = np.array([5, 10, 5])  # 5s, 10s, 5s gaps
        np.testing.assert_array_equal(gaps, expected_gaps)
    
    def test_compute_gap_stats(self):
        """Test gap statistics computation."""
        gaps = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50])
        stats = compute_gap_stats(gaps, [50, 95])
        
        assert stats['gap_p50'] == 5.0
        assert stats['gap_p95'] == 50.0
        assert stats['gap_p95_over_p50'] == 10.0
        assert stats['share_lt_2s'] == 0.1  # 1 out of 10 gaps < 2s
    
    def test_detect_ladder_sweep(self):
        """Test ladder sweep detection."""
        # Consecutive dates
        consecutive_dates = pd.Series([
            '2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18'
        ])
        is_ladder, count = detect_ladder_sweep(consecutive_dates)
        assert is_ladder is True
        assert count > 0
        
        # Non-consecutive dates
        non_consecutive_dates = pd.Series([
            '2025-01-15', '2025-01-20', '2025-01-25'
        ])
        is_ladder, count = detect_ladder_sweep(non_consecutive_dates)
        assert is_ladder is False
        assert count == 0
    
    def test_detect_schedule_regularity(self):
        """Test schedule regularity detection."""
        # Regular schedule (every 15 minutes)
        regular_times = pd.Series([
            datetime(2025, 1, 15, 10, 0, 0),
            datetime(2025, 1, 15, 10, 15, 0),
            datetime(2025, 1, 15, 10, 30, 0),
            datetime(2025, 1, 15, 10, 45, 0),
            datetime(2025, 1, 15, 11, 0, 0)
        ])
        
        is_regular, ratio = detect_schedule_regularity(regular_times)
        assert is_regular is True
        assert ratio == 1.0
        
        # Irregular schedule
        irregular_times = pd.Series([
            datetime(2025, 1, 15, 10, 7, 0),
            datetime(2025, 1, 15, 10, 23, 0),
            datetime(2025, 1, 15, 10, 41, 0),
            datetime(2025, 1, 15, 10, 58, 0)
        ])
        
        is_regular, ratio = detect_schedule_regularity(irregular_times)
        assert is_regular is False
        assert ratio < 0.5


class TestSignatureUtils:
    """Test signature utility functions."""
    
    def test_compute_payload_signature(self):
        """Test payload signature computation."""
        sig1 = compute_payload_signature(
            origin="LAX", destination="NYC", dep_date="2025-01-15",
            ret_date="2025-01-20", pax_config="1-0-0", cabin="economy", stops="0"
        )
        
        sig2 = compute_payload_signature(
            origin="LAX", destination="NYC", dep_date="2025-01-15",
            ret_date="2025-01-20", pax_config="1-0-0", cabin="economy", stops="0"
        )
        
        # Same inputs should produce same signature
        assert sig1 == sig2
        assert len(sig1) == 40  # SHA1 hash length
        
        # Different inputs should produce different signatures
        sig3 = compute_payload_signature(
            origin="LAX", destination="NYC", dep_date="2025-01-16",  # Different date
            ret_date="2025-01-20", pax_config="1-0-0", cabin="economy", stops="0"
        )
        assert sig1 != sig3
    
    def test_compute_signature_stats(self):
        """Test signature statistics computation."""
        signatures = pd.Series([
            'sig1', 'sig1', 'sig1', 'sig2', 'sig2', 'sig3'
        ])
        
        stats = compute_signature_stats(signatures)
        
        assert stats['num_signatures'] == 3
        assert stats['top_signature_share'] == 3/6  # sig1 appears 3 times out of 6
        assert stats['repeat_ratio'] == 3/6  # 3 repeats out of 6 total
    
    def test_compute_route_diversity(self):
        """Test route diversity computation."""
        origins = pd.Series(['LAX', 'LAX', 'SFO', 'SFO'])
        destinations = pd.Series(['NYC', 'NYC', 'NYC', 'CHI'])
        dep_dates = pd.Series(['2025-01-15', '2025-01-16', '2025-01-15', '2025-01-15'])
        
        diversity = compute_route_diversity(origins, destinations, dep_dates)
        
        assert diversity['od_unique'] == 3  # LAX-NYC, SFO-NYC, SFO-CHI
        assert diversity['dep_dates_unique'] == 2  # 2025-01-15, 2025-01-16
    
    def test_detect_pattern_anomalies(self):
        """Test pattern anomaly detection."""
        # High repetition pattern
        high_repetition = pd.Series(['sig1'] * 60 + ['sig2'] * 40)
        anomalies = detect_pattern_anomalies(high_repetition)
        assert anomalies['high_repetition'] is True
        
        # Low diversity pattern
        low_diversity = pd.Series(['sig1'] * 50 + ['sig2'] * 30 + ['sig3'] * 20)
        anomalies = detect_pattern_anomalies(low_diversity)
        assert anomalies['low_diversity'] is False  # 3 signatures is not < 5


class TestStatsUtils:
    """Test statistics utility functions."""
    
    def test_compute_robust_stats(self):
        """Test robust statistics computation."""
        values = np.array([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        
        stats = compute_robust_stats(values)
        
        assert stats['median'] == 3.5  # Median is robust to outliers
        assert stats['mean'] > stats['median']  # Mean is affected by outlier
        assert stats['iqr'] > 0
    
    def test_compute_rate_stats(self):
        """Test rate statistics computation."""
        # Normal case
        rate = compute_rate_stats(3, 10)
        assert rate == 0.3
        
        # Minimum denominator threshold
        rate = compute_rate_stats(1, 5, min_denominator=10)
        assert rate == 0.0
    
    def test_compute_http_version_stats(self):
        """Test HTTP version statistics computation."""
        http_versions = pd.Series(['1.1', '1.1', '2', '2', '2', 'unknown'])
        
        stats = compute_http_version_stats(http_versions)
        
        assert stats['http11_share'] == 2/6
        assert stats['http2_share'] == 3/6
        assert stats['http_unknown_share'] == 1/6
    
    def test_compute_status_code_stats(self):
        """Test status code statistics computation."""
        status_codes = pd.Series([200, 200, 404, 500, 200])
        
        stats = compute_status_code_stats(status_codes)
        
        assert stats['ok2xx_rate'] == 3/5
        assert stats['client4xx_rate'] == 1/5
        assert stats['server5xx_rate'] == 1/5
    
    def test_compute_error_bursts(self):
        """Test error burst computation."""
        # Consecutive errors
        status_codes = pd.Series([200, 404, 404, 404, 200, 500, 500, 200])
        max_burst = compute_error_bursts(status_codes)
        assert max_burst == 3  # 3 consecutive 404s
        
        # No errors
        status_codes = pd.Series([200, 200, 200])
        max_burst = compute_error_bursts(status_codes)
        assert max_burst == 0
    
    def test_compute_hygiene_stats(self):
        """Test hygiene statistics computation."""
        cookies = pd.Series([True, True, False, False, True])
        referrers = pd.Series(['ref1', 'ref2', None, '', 'ref3'])
        
        stats = compute_hygiene_stats(cookies, referrers)
        
        assert stats['cookie_rate'] == 3/5
        assert stats['referrer_rate'] == 4/5  # 4 non-null, non-empty referrers
