"""Statistical utility functions for feature engineering."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats


def compute_robust_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute robust statistics that are less sensitive to outliers."""
    if len(values) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "iqr": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0
        }
    
    # Remove NaN values
    clean_values = values[~np.isnan(values)]
    if len(clean_values) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "iqr": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0
        }
    
    return {
        "mean": np.mean(clean_values),
        "median": np.median(clean_values),
        "std": np.std(clean_values),
        "q25": np.percentile(clean_values, 25),
        "q75": np.percentile(clean_values, 75),
        "iqr": np.percentile(clean_values, 75) - np.percentile(clean_values, 25),
        "skewness": stats.skew(clean_values) if len(clean_values) > 2 else 0.0,
        "kurtosis": stats.kurtosis(clean_values) if len(clean_values) > 3 else 0.0
    }


def compute_rate_stats(
    numerator: int, 
    denominator: int, 
    min_denominator: int = 1
) -> float:
    """Compute rate with minimum denominator threshold."""
    if denominator < min_denominator:
        return 0.0
    return numerator / denominator


def compute_http_version_stats(http_versions: pd.Series) -> Dict[str, float]:
    """Compute HTTP version distribution statistics."""
    if len(http_versions) == 0:
        return {
            "http11_share": 0.0,
            "http2_share": 0.0,
            "http3_share": 0.0,
            "http_unknown_share": 0.0
        }
    
    total = len(http_versions)
    version_counts = http_versions.value_counts()
    
    return {
        "http11_share": version_counts.get("1.1", 0) / total,
        "http2_share": version_counts.get("2", 0) / total,
        "http3_share": version_counts.get("3", 0) / total,
        "http_unknown_share": version_counts.get("unknown", 0) / total
    }


def compute_status_code_stats(status_codes: pd.Series) -> Dict[str, float]:
    """Compute HTTP status code distribution statistics."""
    if len(status_codes) == 0:
        return {
            "ok2xx_rate": 0.0,
            "client4xx_rate": 0.0,
            "server5xx_rate": 0.0
        }
    
    total = len(status_codes)
    
    # Convert to numeric, handling non-numeric values
    numeric_codes = pd.to_numeric(status_codes, errors='coerce')
    
    ok2xx = ((numeric_codes >= 200) & (numeric_codes < 300)).sum()
    client4xx = ((numeric_codes >= 400) & (numeric_codes < 500)).sum()
    server5xx = ((numeric_codes >= 500) & (numeric_codes < 600)).sum()
    
    return {
        "ok2xx_rate": ok2xx / total,
        "client4xx_rate": client4xx / total,
        "server5xx_rate": server5xx / total
    }


def compute_error_bursts(status_codes: pd.Series) -> int:
    """Compute maximum consecutive non-2xx status codes."""
    if len(status_codes) == 0:
        return 0
    
    # Convert to numeric
    numeric_codes = pd.to_numeric(status_codes, errors='coerce')
    
    # Mark non-2xx as errors
    is_error = (numeric_codes < 200) | (numeric_codes >= 300)
    
    # Find consecutive error sequences
    error_sequences = []
    current_sequence = 0
    
    for is_err in is_error:
        if is_err:
            current_sequence += 1
        else:
            if current_sequence > 0:
                error_sequences.append(current_sequence)
            current_sequence = 0
    
    # Add final sequence if it ends with errors
    if current_sequence > 0:
        error_sequences.append(current_sequence)
    
    return max(error_sequences) if error_sequences else 0


def compute_ua_churn_per_ip(ips: pd.Series, user_agents: pd.Series) -> pd.Series:
    """Compute user agent churn per IP address."""
    # Group by IP and count unique user agents
    ua_counts = pd.DataFrame({
        'ip': ips,
        'ua': user_agents
    }).groupby('ip')['ua'].nunique()
    
    # Map back to original series
    return ips.map(ua_counts).fillna(1)


def compute_session_stats(
    session_ids: pd.Series,
    total_events: int
) -> Dict[str, float]:
    """Compute session-related statistics."""
    if len(session_ids) == 0 or total_events == 0:
        return {
            "unique_sessions": 0,
            "sessions_per_event": 0.0
        }
    
    unique_sessions = session_ids.nunique()
    
    return {
        "unique_sessions": unique_sessions,
        "sessions_per_event": unique_sessions / total_events
    }


def compute_hygiene_stats(
    cookies: pd.Series,
    referrers: pd.Series
) -> Dict[str, float]:
    """Compute client hygiene statistics."""
    total = len(cookies) if len(cookies) > 0 else 1
    
    # Cookie rate
    cookie_rate = compute_rate_stats(
        (cookies == True).sum() if cookies.dtype == bool else (cookies == 1).sum(),
        total
    )
    
    # Referrer rate (non-null, non-empty)
    referrer_rate = compute_rate_stats(
        referrers.notna().sum() if len(referrers) > 0 else 0,
        total
    )
    
    return {
        "cookie_rate": cookie_rate,
        "referrer_rate": referrer_rate
    }


def compute_volume_stats(
    n_events: int,
    unique_sessions: int
) -> Dict[str, float]:
    """Compute volume-related statistics."""
    return {
        "n_events": n_events,
        "sessions_per_event": unique_sessions / n_events if n_events > 0 else 0.0
    }


def normalize_features(
    features: pd.DataFrame,
    method: str = "robust"
) -> pd.DataFrame:
    """Normalize features using specified method."""
    from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
    
    if method == "robust":
        scaler = RobustScaler()
    elif method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Only normalize numeric columns
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features_normalized = features.copy()
    
    if len(numeric_cols) > 0:
        features_normalized[numeric_cols] = scaler.fit_transform(features[numeric_cols])
    
    return features_normalized


def compute_correlation_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numeric features."""
    numeric_features = features.select_dtypes(include=[np.number])
    return numeric_features.corr()


def detect_multicollinearity(
    features: pd.DataFrame, 
    threshold: float = 0.95
) -> List[Tuple[str, str, float]]:
    """Detect highly correlated feature pairs."""
    corr_matrix = compute_correlation_matrix(features)
    
    # Find pairs above threshold
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_value
                ))
    
    return high_corr_pairs
