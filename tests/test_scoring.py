"""Tests for scoring module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from botminer.config import Config
from botminer.scoring import (
    score_cohorts, _compute_rule_scores, _compute_anomaly_scores,
    _assign_risk_bands, compute_score_statistics, validate_scoring_results
)


class TestScoring:
    """Test scoring functionality."""
    
    @pytest.fixture
    def sample_features_data(self):
        """Create sample features data for testing."""
        data = {
            'cohort_key': [
                'cohort1', 'cohort2', 'cohort3', 'cohort4', 'cohort5'
            ],
            'date': ['2025-01-15'] * 5,
            'site': ['site1'] * 5,
            'partner': ['partner1'] * 5,
            'ip': [f'192.168.1.{i}' for i in range(1, 6)],
            'user_agent': ['Mozilla/5.0'] * 5,
            'n_events': [100, 50, 200, 30, 150],
            'night_share': [0.8, 0.2, 0.9, 0.1, 0.7],  # High night activity = bot-like
            'hourly_entropy': [1.0, 3.5, 0.5, 4.0, 1.5],  # Low entropy = bot-like
            'top_signature_share': [0.9, 0.3, 0.95, 0.2, 0.8],  # High repetition = bot-like
            'repeat_ratio': [0.8, 0.2, 0.9, 0.1, 0.7],  # High repetition = bot-like
            'gap_p50': [0.5, 30.0, 0.2, 60.0, 1.0],  # Low gaps = bot-like
            'max_run_lt_1s': [20, 2, 50, 1, 15],  # High runs = bot-like
            'cookie_rate': [0.1, 0.9, 0.05, 0.95, 0.2],  # Low cookies = bot-like
            'referrer_rate': [0.2, 0.8, 0.1, 0.9, 0.3],  # Low referrers = bot-like
            'http11_share': [0.9, 0.3, 0.95, 0.2, 0.8],  # High HTTP/1.1 = bot-like
            'http2_share': [0.1, 0.7, 0.05, 0.8, 0.2],
            'ladder_sweep': [True, False, True, False, False],
            'hourly_hist_00': [0.1, 0.01, 0.2, 0.005, 0.15],
            'hourly_hist_01': [0.1, 0.01, 0.2, 0.005, 0.15],
            'hourly_hist_02': [0.1, 0.01, 0.2, 0.005, 0.15],
            'hourly_hist_03': [0.1, 0.01, 0.2, 0.005, 0.15],
            'hourly_hist_04': [0.1, 0.01, 0.2, 0.005, 0.15],
            'hourly_hist_05': [0.1, 0.01, 0.2, 0.005, 0.15],
            'hourly_hist_06': [0.1, 0.01, 0.2, 0.005, 0.15],
            'hourly_hist_07': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_08': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_09': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_10': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_11': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_12': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_13': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_14': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_15': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_16': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_17': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_18': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_19': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_20': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_21': [0.01, 0.1, 0.005, 0.15, 0.02],
            'hourly_hist_22': [0.1, 0.01, 0.2, 0.005, 0.15],
            'hourly_hist_23': [0.1, 0.01, 0.2, 0.005, 0.15]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config.from_file("config.yaml")
    
    def test_compute_rule_scores(self, sample_features_data, config):
        """Test rule-based scoring computation."""
        rule_scores = _compute_rule_scores(sample_features_data, config)
        
        # Check that scores are in [0, 1] range
        assert np.all(rule_scores >= 0)
        assert np.all(rule_scores <= 1)
        
        # Check that bot-like cohorts get higher scores
        # cohort1 and cohort3 should have high scores (bot-like)
        # cohort2 and cohort4 should have low scores (human-like)
        assert rule_scores[0] > rule_scores[1]  # cohort1 > cohort2
        assert rule_scores[2] > rule_scores[3]  # cohort3 > cohort4
    
    def test_compute_anomaly_scores(self, sample_features_data, config):
        """Test anomaly scoring computation."""
        iso_scores = _compute_anomaly_scores(sample_features_data, config)
        
        # Check that scores are in [0, 1] range
        assert np.all(iso_scores >= 0)
        assert np.all(iso_scores <= 1)
        
        # Check that we have some variation in scores
        assert np.std(iso_scores) > 0
    
    def test_assign_risk_bands(self, config):
        """Test risk band assignment."""
        botness_scores = pd.Series([0.9, 0.6, 0.3, 0.1])
        risk_bands = _assign_risk_bands(botness_scores, config)
        
        expected_bands = ['High', 'Medium', 'Low', 'Low']
        assert list(risk_bands) == expected_bands
    
    def test_score_cohorts_full_pipeline(self, sample_features_data, config):
        """Test full scoring pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save sample features
            features_path = os.path.join(temp_dir, '2025-01-15.parquet')
            sample_features_data.to_parquet(features_path, index=False)
            
            # Update config paths
            config.io.features_dir = temp_dir
            
            # Run scoring
            result = score_cohorts('2025-01-15', config)
            
            # Check results
            assert result['cohorts_scored'] == 5
            assert 'high_risk' in result
            assert 'medium_risk' in result
            assert 'low_risk' in result
            assert result['high_risk'] + result['medium_risk'] + result['low_risk'] == 5
            
            # Check output file exists
            output_path = Path(result['output_path'])
            assert output_path.exists()
            
            # Load and verify scored features
            scored_df = pd.read_parquet(output_path)
            assert len(scored_df) == 5
            assert 'rule_score' in scored_df.columns
            assert 'iso_score' in scored_df.columns
            assert 'botness' in scored_df.columns
            assert 'risk_band' in scored_df.columns
            
            # Check that scores are properly combined
            expected_botness = (
                config.scoring.alpha_blend_iso * scored_df['rule_score'] +
                (1 - config.scoring.alpha_blend_iso) * scored_df['iso_score']
            )
            np.testing.assert_array_almost_equal(scored_df['botness'], expected_botness)
    
    def test_compute_score_statistics(self, sample_features_data, config):
        """Test score statistics computation."""
        # Add mock scores
        sample_features_data['rule_score'] = [0.8, 0.3, 0.9, 0.2, 0.6]
        sample_features_data['iso_score'] = [0.7, 0.4, 0.8, 0.3, 0.5]
        sample_features_data['botness'] = [0.75, 0.35, 0.85, 0.25, 0.55]
        sample_features_data['risk_band'] = ['High', 'Low', 'High', 'Low', 'Medium']
        
        stats = compute_score_statistics(sample_features_data)
        
        # Check structure
        assert 'total_cohorts' in stats
        assert 'risk_distribution' in stats
        assert 'score_statistics' in stats
        
        # Check values
        assert stats['total_cohorts'] == 5
        assert stats['risk_distribution']['High'] == 2
        assert stats['risk_distribution']['Medium'] == 1
        assert stats['risk_distribution']['Low'] == 2
        
        # Check score statistics
        assert 'rule_score' in stats['score_statistics']
        assert 'iso_score' in stats['score_statistics']
        assert 'botness' in stats['score_statistics']
    
    def test_validate_scoring_results(self, sample_features_data):
        """Test scoring results validation."""
        # Valid data
        valid_data = sample_features_data.copy()
        valid_data['rule_score'] = [0.8, 0.3, 0.9, 0.2, 0.6]
        valid_data['iso_score'] = [0.7, 0.4, 0.8, 0.3, 0.5]
        valid_data['botness'] = [0.75, 0.35, 0.85, 0.25, 0.55]
        valid_data['risk_band'] = ['High', 'Low', 'High', 'Low', 'Medium']
        
        issues = validate_scoring_results(valid_data)
        assert len(issues) == 0
        
        # Invalid data - missing scores
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'rule_score'] = np.nan
        
        issues = validate_scoring_results(invalid_data)
        assert len(issues) > 0
        assert any('missing rule scores' in issue for issue in issues)
        
        # Invalid data - out of range scores
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'botness'] = 1.5  # > 1.0
        
        issues = validate_scoring_results(invalid_data)
        assert len(issues) > 0
        assert any('not in [0, 1] range' in issue for issue in issues)
        
        # Invalid data - invalid risk bands
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'risk_band'] = 'Invalid'
        
        issues = validate_scoring_results(invalid_data)
        assert len(issues) > 0
        assert any('Invalid risk bands' in issue for issue in issues)
    
    def test_scoring_with_missing_features(self, config):
        """Test scoring with missing features."""
        # Create data with missing features
        incomplete_data = {
            'cohort_key': ['cohort1', 'cohort2'],
            'date': ['2025-01-15'] * 2,
            'site': ['site1'] * 2,
            'partner': ['partner1'] * 2,
            'ip': ['192.168.1.1', '192.168.1.2'],
            'user_agent': ['Mozilla/5.0'] * 2,
            'n_events': [100, 50],
            # Missing many features
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            features_path = os.path.join(temp_dir, '2025-01-15.parquet')
            pd.DataFrame(incomplete_data).to_parquet(features_path, index=False)
            
            config.io.features_dir = temp_dir
            
            # Should handle missing features gracefully
            result = score_cohorts('2025-01-15', config)
            assert result['cohorts_scored'] == 2
    
    def test_scoring_edge_cases(self, config):
        """Test scoring with edge cases."""
        # Create data with edge cases
        edge_case_data = {
            'cohort_key': ['edge_cohort'],
            'date': ['2025-01-15'],
            'site': ['site1'],
            'partner': ['partner1'],
            'ip': ['192.168.1.1'],
            'user_agent': ['Mozilla/5.0'],
            'n_events': [1],  # Very low volume
            'night_share': [0.0],  # No night activity
            'hourly_entropy': [0.0],  # No entropy
            'top_signature_share': [1.0],  # Perfect repetition
            'repeat_ratio': [0.0],  # No repeats
            'gap_p50': [0.0],  # No gaps
            'max_run_lt_1s': [0],  # No runs
            'cookie_rate': [1.0],  # Perfect cookies
            'referrer_rate': [1.0],  # Perfect referrers
            'http11_share': [0.0],  # No HTTP/1.1
            'http2_share': [1.0],  # Perfect HTTP/2
            'ladder_sweep': [False],
            # Add hourly histogram
            **{f'hourly_hist_{i:02d}': [0.0] for i in range(24)}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            features_path = os.path.join(temp_dir, '2025-01-15.parquet')
            pd.DataFrame(edge_case_data).to_parquet(features_path, index=False)
            
            config.io.features_dir = temp_dir
            
            # Should handle edge cases without errors
            result = score_cohorts('2025-01-15', config)
            assert result['cohorts_scored'] == 1
