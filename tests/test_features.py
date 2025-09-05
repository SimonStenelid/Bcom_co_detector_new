"""Tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from botminer.config import Config
from botminer.features import build_features, _build_cohort_features


class TestFeatures:
    """Test feature engineering functionality."""
    
    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data for testing."""
        data = {
            'cohort_key': [
                '2025-01-15|site1|partner1|192.168.1.1|Mozilla/5.0',
                '2025-01-15|site1|partner1|192.168.1.1|Mozilla/5.0',
                '2025-01-15|site1|partner1|192.168.1.1|Mozilla/5.0',
                '2025-01-15|site1|partner1|192.168.1.1|Mozilla/5.0',
                '2025-01-15|site1|partner1|192.168.1.1|Mozilla/5.0',
                '2025-01-15|site1|partner1|192.168.1.2|Mozilla/5.0',
                '2025-01-15|site1|partner1|192.168.1.2|Mozilla/5.0',
                '2025-01-15|site1|partner1|192.168.1.2|Mozilla/5.0',
                '2025-01-15|site1|partner1|192.168.1.2|Mozilla/5.0',
                '2025-01-15|site1|partner1|192.168.1.2|Mozilla/5.0'
            ],
            'date': ['2025-01-15'] * 10,
            'site': ['site1'] * 10,
            'partner': ['partner1'] * 10,
            'ip': ['192.168.1.1'] * 5 + ['192.168.1.2'] * 5,
            'user_agent': ['Mozilla/5.0'] * 10,
            'timestamp': pd.date_range('2025-01-15 10:00:00', periods=10, freq='1min'),
            'hour': [10] * 10,
            'minute': list(range(10)),
            'day_of_week': [2] * 10,  # Wednesday
            'payload_signature': [
                'sig1', 'sig1', 'sig1', 'sig2', 'sig2',
                'sig1', 'sig1', 'sig2', 'sig2', 'sig3'
            ],
            'origin': ['LAX'] * 5 + ['SFO'] * 5,
            'destination': ['NYC'] * 10,
            'dep_date': [
                '2025-01-20', '2025-01-20', '2025-01-20', '2025-01-21', '2025-01-21',
                '2025-01-20', '2025-01-20', '2025-01-21', '2025-01-21', '2025-01-22'
            ],
            'ret_date': [''] * 10,
            'pax_config': ['1-0-0'] * 10,
            'cabin': ['economy'] * 10,
            'stops': ['0'] * 10,
            'session_id': ['sess1'] * 5 + ['sess2'] * 5,
            'referrer': ['https://google.com'] * 5 + [''] * 5,
            'cookie_present': [True] * 5 + [False] * 5,
            'http_version': ['1.1'] * 5 + ['2'] * 5,
            'status_code': [200] * 8 + [404] * 2
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config.from_file("config.yaml")
    
    def test_build_cohort_features(self, sample_raw_data, config):
        """Test building features for a single cohort."""
        # Get first cohort
        cohort_data = sample_raw_data[sample_raw_data['cohort_key'] == sample_raw_data['cohort_key'].iloc[0]]
        
        features = _build_cohort_features(cohort_data, config)
        
        # Check basic cohort info
        assert features['cohort_key'] == sample_raw_data['cohort_key'].iloc[0]
        assert features['site'] == 'site1'
        assert features['partner'] == 'partner1'
        assert features['ip'] == '192.168.1.1'
        
        # Check time features
        assert 'hourly_hist_00' in features
        assert 'hourly_hist_23' in features
        assert 'hourly_entropy' in features
        assert 'night_share' in features
        assert 'gap_p50' in features
        assert 'gap_p95' in features
        
        # Check signature features
        assert 'num_signatures' in features
        assert 'top_signature_share' in features
        assert 'repeat_ratio' in features
        assert 'od_unique' in features
        assert 'dep_dates_unique' in features
        
        # Check client features
        assert 'http11_share' in features
        assert 'http2_share' in features
        assert 'cookie_rate' in features
        assert 'referrer_rate' in features
        
        # Check volume features
        assert 'n_events' in features
        assert 'unique_sessions' in features
        assert 'sessions_per_event' in features
        
        # Check outcome features
        assert 'ok2xx_rate' in features
        assert 'client4xx_rate' in features
        assert 'server5xx_rate' in features
        assert 'error_bursts' in features
    
    def test_build_features_full_pipeline(self, sample_raw_data, config):
        """Test full feature building pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save sample data as parquet
            raw_path = os.path.join(temp_dir, '2025-01-15.parquet')
            sample_raw_data.to_parquet(raw_path, index=False)
            
            # Update config paths
            config.io.input_dir = temp_dir
            config.io.features_dir = temp_dir
            config.ingest.min_events_per_cohort = 3  # Lower threshold for testing
            
            # Build features
            result = build_features('2025-01-15', config)
            
            # Check results
            assert result['cohorts_processed'] == 2  # 2 cohorts with >= 3 events
            assert 'output_path' in result
            
            # Check output file exists
            output_path = Path(result['output_path'])
            assert output_path.exists()
            
            # Load and verify output
            features_df = pd.read_parquet(output_path)
            assert len(features_df) == 2
            assert 'cohort_key' in features_df.columns
            assert 'rule_score' not in features_df.columns  # Not scored yet
    
    def test_feature_engineering_edge_cases(self, config):
        """Test feature engineering with edge cases."""
        # Create data with edge cases
        edge_case_data = {
            'cohort_key': ['edge_cohort'] * 2,
            'date': ['2025-01-15'] * 2,
            'site': ['site1'] * 2,
            'partner': ['partner1'] * 2,
            'ip': ['192.168.1.1'] * 2,
            'user_agent': ['Mozilla/5.0'] * 2,
            'timestamp': pd.Series([
                pd.Timestamp('2025-01-15 10:00:00'),
                pd.Timestamp('2025-01-15 10:00:00')  # Same timestamp
            ]),
            'hour': [10] * 2,
            'minute': [0] * 2,
            'day_of_week': [2] * 2,
            'payload_signature': ['sig1', 'sig1'],  # Same signature
            'origin': ['LAX'] * 2,
            'destination': ['NYC'] * 2,
            'dep_date': ['2025-01-20'] * 2,
            'ret_date': [''] * 2,
            'pax_config': ['1-0-0'] * 2,
            'cabin': ['economy'] * 2,
            'stops': ['0'] * 2,
            'session_id': ['sess1'] * 2,
            'referrer': [''] * 2,
            'cookie_present': [False] * 2,
            'http_version': ['unknown'] * 2,
            'status_code': [500] * 2  # All errors
        }
        
        edge_df = pd.DataFrame(edge_case_data)
        features = _build_cohort_features(edge_df, config)
        
        # Check that edge cases are handled gracefully
        assert features['n_events'] == 2
        assert features['gap_p50'] == 0.0  # Same timestamps
        assert features['top_signature_share'] == 1.0  # All same signature
        assert features['ok2xx_rate'] == 0.0  # All errors
        assert features['server5xx_rate'] == 1.0  # All 5xx errors
    
    def test_historical_features_handling(self, sample_raw_data, config):
        """Test handling of historical features when no history exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save sample data
            raw_path = os.path.join(temp_dir, '2025-01-15.parquet')
            sample_raw_data.to_parquet(raw_path, index=False)
            
            config.io.input_dir = temp_dir
            config.io.features_dir = temp_dir
            config.io.history_dir = temp_dir
            config.ingest.min_events_per_cohort = 3
            
            # Build features (no historical data should exist)
            result = build_features('2025-01-15', config)
            
            # Should still work without historical data
            assert result['cohorts_processed'] == 2
            
            # Load features and check no historical columns
            features_df = pd.read_parquet(result['output_path'])
            historical_cols = [col for col in features_df.columns if col.startswith('ewma_')]
            assert len(historical_cols) == 0  # No historical features added
    
    def test_minimum_events_filtering(self, sample_raw_data, config):
        """Test filtering by minimum events per cohort."""
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = os.path.join(temp_dir, '2025-01-15.parquet')
            sample_raw_data.to_parquet(raw_path, index=False)
            
            config.io.input_dir = temp_dir
            config.io.features_dir = temp_dir
            config.ingest.min_events_per_cohort = 10  # High threshold
            
            # Should filter out cohorts with < 10 events
            result = build_features('2025-01-15', config)
            assert result['cohorts_processed'] == 0  # No cohorts meet threshold
            
            # Lower threshold
            config.ingest.min_events_per_cohort = 3
            result = build_features('2025-01-15', config)
            assert result['cohorts_processed'] == 2  # Both cohorts meet threshold
