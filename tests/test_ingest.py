"""Tests for ingestion module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from botminer.config import Config
from botminer.ingest import ingest_csv, _normalize_dataframe, _add_derived_columns


class TestIngest:
    """Test ingestion functionality."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        data = {
            'timestamp': [
                '2025-01-15T10:00:00Z',
                '2025-01-15T10:01:00Z',
                '2025-01-15T10:02:00Z'
            ],
            'site': ['site1', 'site1', 'site1'],
            'partner': ['partner1', 'partner1', 'partner1'],
            'origin': ['LAX', 'LAX', 'SFO'],
            'destination': ['NYC', 'NYC', 'NYC'],
            'journey_type': ['OW', 'OW', 'RT'],
            'dep_date': ['2025-01-20', '2025-01-21', '2025-01-22'],
            'ret_date': ['', '', '2025-01-25'],
            'pax_config': ['1-0-0', '1-0-0', '2-0-0'],
            'cabin': ['economy', 'economy', 'business'],
            'stops': ['0', '0', '1'],
            'ip': ['192.168.1.1', '192.168.1.1', '192.168.1.2'],
            'user_agent': ['Mozilla/5.0', 'Mozilla/5.0', 'Mozilla/5.0'],
            'session_id': ['sess1', 'sess1', 'sess2'],
            'referrer': ['https://google.com', '', 'https://bing.com'],
            'cookie_present': [True, False, True],
            'http_version': ['1.1', '2', '1.1'],
            'status_code': [200, 200, 404],
            'request_id': ['req1', 'req2', 'req3']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config.from_file("config.yaml")
    
    def test_normalize_dataframe(self, sample_csv_data, config):
        """Test dataframe normalization."""
        normalized_df = _normalize_dataframe(sample_csv_data, config)
        
        # Check timestamp normalization
        assert 'timestamp' in normalized_df.columns
        assert normalized_df['timestamp'].dtype == 'datetime64[ns, UTC]'
        
        # Check string normalization
        assert normalized_df['site'].dtype == 'object'
        assert normalized_df['site'].iloc[0] == 'site1'
        
        # Check boolean normalization
        assert normalized_df['cookie_present'].dtype == 'bool'
        assert normalized_df['cookie_present'].iloc[0] == True
    
    def test_add_derived_columns(self, sample_csv_data, config):
        """Test derived column addition."""
        normalized_df = _normalize_dataframe(sample_csv_data, config)
        derived_df = _add_derived_columns(normalized_df, '2025-01-15', config)
        
        # Check cohort key creation
        assert 'cohort_key' in derived_df.columns
        assert 'cohort_date' in derived_df.columns
        assert 'cohort_site' in derived_df.columns
        
        # Check time extraction
        assert 'hour' in derived_df.columns
        assert 'minute' in derived_df.columns
        assert 'day_of_week' in derived_df.columns
        
        # Check payload signature
        assert 'payload_signature' in derived_df.columns
        assert len(derived_df['payload_signature'].iloc[0]) == 40  # SHA1 length
        
        # Check cohort key uniqueness
        unique_cohorts = derived_df['cohort_key'].nunique()
        assert unique_cohorts == 2  # 2 different IPs
    
    def test_ingest_csv_full_pipeline(self, sample_csv_data, config):
        """Test full CSV ingestion pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary CSV file
            csv_path = os.path.join(temp_dir, 'test_data.csv')
            sample_csv_data.to_csv(csv_path, index=False)
            
            # Update config to use temp directory
            config.io.input_dir = temp_dir
            
            # Run ingestion
            result = ingest_csv(csv_path, '2025-01-15', config)
            
            # Check results
            assert result['rows_processed'] == 3
            assert result['date'] == '2025-01-15'
            assert 'output_path' in result
            
            # Check output file exists
            output_path = Path(result['output_path'])
            assert output_path.exists()
            
            # Load and verify output
            output_df = pd.read_parquet(output_path)
            assert len(output_df) == 3
            assert 'cohort_key' in output_df.columns
            assert 'payload_signature' in output_df.columns
    
    def test_ingest_missing_columns(self, config):
        """Test ingestion with missing columns."""
        # Create data with missing required columns
        incomplete_data = {
            'timestamp': ['2025-01-15T10:00:00Z'],
            'site': ['site1'],
            'ip': ['192.168.1.1']
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'incomplete_data.csv')
            pd.DataFrame(incomplete_data).to_csv(csv_path, index=False)
            
            config.io.input_dir = temp_dir
            
            # Should not raise error, but add missing columns
            result = ingest_csv(csv_path, '2025-01-15', config)
            assert result['rows_processed'] == 1
    
    def test_ingest_empty_file(self, config):
        """Test ingestion with empty CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'empty_data.csv')
            
            # Create empty CSV with header
            pd.DataFrame(columns=['timestamp', 'site', 'partner']).to_csv(csv_path, index=False)
            
            config.io.input_dir = temp_dir
            
            result = ingest_csv(csv_path, '2025-01-15', config)
            assert result['rows_processed'] == 0
    
    def test_ingest_invalid_timestamps(self, config):
        """Test ingestion with invalid timestamps."""
        data_with_bad_timestamps = {
            'timestamp': ['invalid', '2025-01-15T10:00:00Z', ''],
            'site': ['site1', 'site1', 'site1'],
            'partner': ['partner1', 'partner1', 'partner1'],
            'origin': ['LAX', 'LAX', 'LAX'],
            'destination': ['NYC', 'NYC', 'NYC'],
            'journey_type': ['OW', 'OW', 'OW'],
            'dep_date': ['2025-01-20', '2025-01-21', '2025-01-22'],
            'ret_date': ['', '', ''],
            'pax_config': ['1-0-0', '1-0-0', '1-0-0'],
            'cabin': ['economy', 'economy', 'economy'],
            'stops': ['0', '0', '0'],
            'ip': ['192.168.1.1', '192.168.1.1', '192.168.1.1'],
            'user_agent': ['Mozilla/5.0', 'Mozilla/5.0', 'Mozilla/5.0']
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'bad_timestamps.csv')
            pd.DataFrame(data_with_bad_timestamps).to_csv(csv_path, index=False)
            
            config.io.input_dir = temp_dir
            
            # Should handle invalid timestamps gracefully
            result = ingest_csv(csv_path, '2025-01-15', config)
            assert result['rows_processed'] == 3
            
            # Check that timestamps were processed
            output_path = Path(result['output_path'])
            output_df = pd.read_parquet(output_path)
            assert 'timestamp' in output_df.columns
