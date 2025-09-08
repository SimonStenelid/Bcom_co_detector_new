"""Scoring module for botness detection using rule-based and anomaly detection."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import pickle
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

from .config import Config


def score_cohorts(date: str, config: Config) -> Dict[str, Any]:
    """
    Score cohorts for botness using rule-based and anomaly detection.
    
    Args:
        date: Date string in YYYY-MM-DD format
        config: Configuration object
        
    Returns:
        Dictionary with scoring results
    """
    logger = logging.getLogger(__name__)
    
    # Load features
    features_path = config.get_paths(date)['features']
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    
    logger.info(f"Loading features from: {features_path}")
    features_df = pd.read_parquet(features_path)
    
    logger.info(f"Scoring {len(features_df)} cohorts")
    
    # Compute rule-based scores
    rule_scores = _compute_rule_scores(features_df, config)
    features_df['rule_score'] = rule_scores
    
    # Compute anomaly scores using Isolation Forest
    iso_scores = _compute_anomaly_scores(features_df, config, date)
    features_df['iso_score'] = iso_scores
    
    # Combine scores
    features_df['botness'] = (
        config.scoring.alpha_blend_iso * features_df['rule_score'] +
        (1 - config.scoring.alpha_blend_iso) * features_df['iso_score']
    )
    
    # Assign risk bands
    features_df['risk_band'] = _assign_risk_bands(features_df['botness'], config)
    
    # Sort by botness score
    features_df = features_df.sort_values('botness', ascending=False)
    
    # Save scored features
    output_path = config.get_paths(date)['features']  # Overwrite with scores
    logger.info(f"Saving scored features to: {output_path}")
    features_df.to_parquet(output_path, index=False, compression='snappy')
    
    # Count risk bands
    risk_counts = features_df['risk_band'].value_counts()
    
    return {
        'cohorts_scored': len(features_df),
        'high_risk': risk_counts.get('High', 0),
        'medium_risk': risk_counts.get('Medium', 0),
        'low_risk': risk_counts.get('Low', 0),
        'output_path': str(output_path)
    }


def _compute_rule_scores(features_df: pd.DataFrame, config: Config) -> np.ndarray:
    """Compute rule-based botness scores."""
    logger = logging.getLogger(__name__)
    
    scores = np.zeros(len(features_df))
    weights = config.scoring.weights
    
    # Cadence score (high night share + low entropy = more bot-like)
    if 'night_share' in features_df.columns and 'hourly_entropy' in features_df.columns:
        cadence_score = (
            features_df['night_share'].fillna(0) +
            (1 - features_df['hourly_entropy'].fillna(0) / 4.0)  # Normalize entropy
        ) / 2.0
        scores += weights['cadence'] * cadence_score
    
    # Repetition score (high signature repetition = more bot-like)
    if 'top_signature_share' in features_df.columns and 'repeat_ratio' in features_df.columns:
        repetition_score = (
            features_df['top_signature_share'].fillna(0) +
            features_df['repeat_ratio'].fillna(0)
        ) / 2.0
        scores += weights['repetition'] * repetition_score
    
    # Speed score (low gaps + long runs = more bot-like)
    if 'gap_p50' in features_df.columns and 'max_run_lt_1s' in features_df.columns:
        # Normalize gap_p50 (lower is more bot-like)
        gap_score = 1 - np.minimum(features_df['gap_p50'].fillna(0) / 60.0, 1.0)
        # Normalize max_run_lt_1s (higher is more bot-like)
        run_score = np.minimum(features_df['max_run_lt_1s'].fillna(0) / 10.0, 1.0)
        speed_score = (gap_score + run_score) / 2.0
        scores += weights['speed'] * speed_score
    
    # Hygiene score (low cookie/referrer rates = more bot-like)
    if 'cookie_rate' in features_df.columns and 'referrer_rate' in features_df.columns:
        hygiene_score = (
            (1 - features_df['cookie_rate'].fillna(0)) +
            (1 - features_df['referrer_rate'].fillna(0))
        ) / 2.0
        scores += weights['hygiene'] * hygiene_score
    
    # Protocol score (suspicious HTTP patterns = more bot-like)
    if 'http11_share' in features_df.columns and 'http2_share' in features_df.columns:
        # High HTTP/1.1 or "too perfect" HTTP/2
        protocol_score = np.maximum(
            features_df['http11_share'].fillna(0),
            (features_df['http2_share'].fillna(0) > 0.95).astype(float)
        )
        scores += weights['protocol'] * protocol_score
    
    # Volume score (high volume with other bot traits = more bot-like)
    if 'n_events' in features_df.columns:
        # Normalize volume (log scale)
        volume_score = np.minimum(
            np.log1p(features_df['n_events'].fillna(0)) / 10.0, 1.0
        )
        scores += weights['volume'] * volume_score

    # IP churn score (many user agents per IP = more bot-like)
    if 'ua_churn_per_ip_day' in features_df.columns and 'ip_churn' in weights:
        ip_score = np.minimum(
            (features_df['ua_churn_per_ip_day'].fillna(1) - 1) / 4.0,
            1.0
        )
        scores += weights['ip_churn'] * ip_score
    
    # Normalize to [0, 1]
    scores = np.clip(scores, 0, 1)
    
    logger.info(f"Rule scores - mean: {scores.mean():.3f}, std: {scores.std():.3f}")
    
    return scores


def _compute_anomaly_scores(features_df: pd.DataFrame, config: Config, date: str = None) -> np.ndarray:
    """Compute anomaly scores using Isolation Forest."""
    logger = logging.getLogger(__name__)
    
    # Select numeric features for anomaly detection
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    
    # Remove non-feature columns
    exclude_cols = [
        'rule_score', 'iso_score', 'botness', 'risk_band',
        'cohort_key', 'date', 'site', 'partner', 'ip', 'user_agent'
    ]
    
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(feature_cols) == 0:
        logger.warning("No numeric features found for anomaly detection")
        return np.zeros(len(features_df))

    if len(features_df) < 2:
        logger.warning("Not enough samples for anomaly detection")
        return np.zeros(len(features_df))
    
    # Prepare feature matrix
    X = features_df[feature_cols].fillna(0)
    
    # Normalize features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=config.isolation_forest.n_estimators,
        contamination=config.isolation_forest.contamination,
        random_state=config.isolation_forest.random_state,
        max_samples=config.isolation_forest.max_samples
    )
    
    iso_forest.fit(X_scaled)
    
    # Get anomaly scores
    anomaly_scores = iso_forest.decision_function(X_scaled)
    
    # Convert to [0, 1] scale (higher = more anomalous)
    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()
    
    if max_score > min_score:
        normalized_scores = (anomaly_scores - min_score) / (max_score - min_score)
    else:
        normalized_scores = np.zeros_like(anomaly_scores)
    
    logger.info(f"Anomaly scores - mean: {normalized_scores.mean():.3f}, std: {normalized_scores.std():.3f}")
    
    # Save model for future use
    if date is not None:
        model_path = config.get_paths(date)['features'].parent / f"iso_model_{date}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': iso_forest,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'date': date
            }, f)
        logger.info(f"Saved Isolation Forest model to: {model_path}")
    
    return normalized_scores


def _assign_risk_bands(botness_scores: pd.Series, config: Config) -> pd.Series:
    """Assign risk bands based on botness scores."""
    thresholds = config.scoring.thresholds
    
    def assign_band(score):
        if score >= thresholds['high']:
            return 'High'
        elif score >= thresholds['medium']:
            return 'Medium'
        else:
            return 'Low'
    
    return botness_scores.apply(assign_band)


def load_scored_features(date: str, config: Config) -> pd.DataFrame:
    """Load scored features for a given date."""
    features_path = config.get_paths(date)['features']
    
    if not features_path.exists():
        raise FileNotFoundError(f"Scored features not found: {features_path}")
    
    return pd.read_parquet(features_path)


def get_top_risk_cohorts(
    features_df: pd.DataFrame, 
    risk_band: str = 'High',
    top_n: int = 50
) -> pd.DataFrame:
    """Get top N cohorts for a given risk band."""
    filtered_df = features_df[features_df['risk_band'] == risk_band]
    return filtered_df.head(top_n)


def compute_score_statistics(features_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute statistics about the scoring results."""
    stats = {
        'total_cohorts': len(features_df),
        'risk_distribution': features_df['risk_band'].value_counts().to_dict(),
        'score_statistics': {
            'rule_score': {
                'mean': features_df['rule_score'].mean(),
                'std': features_df['rule_score'].std(),
                'min': features_df['rule_score'].min(),
                'max': features_df['rule_score'].max()
            },
            'iso_score': {
                'mean': features_df['iso_score'].mean(),
                'std': features_df['iso_score'].std(),
                'min': features_df['iso_score'].min(),
                'max': features_df['iso_score'].max()
            },
            'botness': {
                'mean': features_df['botness'].mean(),
                'std': features_df['botness'].std(),
                'min': features_df['botness'].min(),
                'max': features_df['botness'].max()
            }
        }
    }
    
    return stats


def validate_scoring_results(features_df: pd.DataFrame) -> List[str]:
    """Validate scoring results and return any issues found."""
    issues = []
    
    # Check for missing scores
    if features_df['rule_score'].isna().any():
        issues.append("Some cohorts have missing rule scores")
    
    if features_df['iso_score'].isna().any():
        issues.append("Some cohorts have missing anomaly scores")
    
    if features_df['botness'].isna().any():
        issues.append("Some cohorts have missing botness scores")
    
    # Check for invalid risk bands
    valid_bands = {'High', 'Medium', 'Low'}
    invalid_bands = set(features_df['risk_band'].unique()) - valid_bands
    if invalid_bands:
        issues.append(f"Invalid risk bands found: {invalid_bands}")
    
    # Check score ranges
    if not (0 <= features_df['rule_score'].min() <= features_df['rule_score'].max() <= 1):
        issues.append("Rule scores are not in [0, 1] range")
    
    if not (0 <= features_df['iso_score'].min() <= features_df['iso_score'].max() <= 1):
        issues.append("Anomaly scores are not in [0, 1] range")
    
    if not (0 <= features_df['botness'].min() <= features_df['botness'].max() <= 1):
        issues.append("Botness scores are not in [0, 1] range")
    
    return issues
