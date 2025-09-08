"""Time-related utility functions."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime, timezone


def parse_timestamp(ts_str: str, tz: str = "UTC") -> pd.Timestamp:
    """Parse timestamp string to pandas Timestamp with timezone."""
    try:
        # Try parsing ISO format first
        if 'T' in ts_str or 'Z' in ts_str:
            ts = pd.to_datetime(ts_str, utc=True)
        else:
            # Assume it's a simple format
            ts = pd.to_datetime(ts_str)
            if ts.tz is None:
                ts = ts.tz_localize(tz)
        
        return ts
    except Exception:
        # Fallback to current time if parsing fails
        return pd.Timestamp.now(tz=tz)


def extract_hour(timestamps: pd.Series) -> pd.Series:
    """Extract hour from timestamps."""
    return timestamps.dt.hour


def is_night_time(hours: pd.Series, night_hours: List[int]) -> pd.Series:
    """Check if hours fall within night time range."""
    return hours.isin(night_hours)


def compute_hourly_histogram(hours: pd.Series) -> np.ndarray:
    """Compute hourly histogram as proportions."""
    hist, _ = np.histogram(hours, bins=24, range=(0, 24), density=True)
    return hist


def compute_hourly_entropy(hourly_hist: np.ndarray) -> float:
    """Compute entropy of hourly distribution."""
    hist = hourly_hist[hourly_hist > 0]
    if len(hist) == 0:
        return 0.0
    return float(-np.sum(hist * np.log2(hist)))


def compute_gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient for inequality measurement."""
    if len(values) == 0:
        return 0.0
    
    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Compute Gini coefficient
    cumsum = np.cumsum(sorted_values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0


def compute_time_gaps(timestamps: pd.Series) -> np.ndarray:
    """Compute time gaps between consecutive events."""
    if len(timestamps) < 2:
        return np.array([])
    
    sorted_times = timestamps.sort_values()
    gaps = sorted_times.diff().dropna().dt.total_seconds()
    return gaps.values


def compute_gap_stats(gaps: np.ndarray, percentiles: List[int]) -> dict:
    """Compute statistics for time gaps."""
    if len(gaps) == 0:
        return {f"gap_p{p}": 0.0 for p in percentiles}
    
    stats = {}
    for p in percentiles:
        method = 'higher' if p >= 95 else 'lower'
        stats[f"gap_p{p}"] = float(np.percentile(gaps, p, method=method))
    
    # Additional gap statistics
    stats["gap_p95_over_p50"] = (
        stats["gap_p95"] / stats["gap_p50"] 
        if stats["gap_p50"] > 0 else 0.0
    )
    
    # Count of gaps less than threshold
    stats["max_run_lt_1s"] = np.max(np.convolve(
        (gaps < 1.0).astype(int), 
        np.ones(10, dtype=int), 
        mode='valid'
    )) if len(gaps) >= 10 else 0
    
    stats["share_lt_2s"] = np.mean(gaps < 2.0)
    
    return stats


def detect_ladder_sweep(
    dep_dates: pd.Series, 
    max_gap_days: int = 3
) -> Tuple[bool, int]:
    """Detect ladder sweep pattern in departure dates."""
    if len(dep_dates) < 3:
        return False, 0
    
    try:
        # Convert to datetime if needed
        dates = pd.to_datetime(dep_dates).sort_values()
        
        # Find consecutive date sequences
        date_diffs = dates.diff().dt.days.dropna()
        
        # Count sequences where difference is 1 day
        consecutive_count = 0
        current_sequence = 0
        
        for diff in date_diffs:
            if diff == 1:
                current_sequence += 1
            else:
                if current_sequence >= 2:  # At least 3 consecutive dates
                    consecutive_count += current_sequence
                current_sequence = 0
        
        # Check final sequence
        if current_sequence >= 2:
            consecutive_count += current_sequence
        
        return consecutive_count > 0, consecutive_count
        
    except Exception:
        return False, 0


def compute_ewma(
    values: pd.Series, 
    alpha: float, 
    min_periods: int = 1
) -> pd.Series:
    """Compute exponentially weighted moving average."""
    return values.ewm(alpha=alpha, min_periods=min_periods).mean()


def compute_jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def detect_schedule_regularity(timestamps: pd.Series) -> Tuple[bool, float]:
    """Detect if timestamps follow a regular schedule (e.g., :00, :15, :30, :45)."""
    if len(timestamps) < 5:
        return False, 0.0
    
    # Extract minutes
    minutes = timestamps.dt.minute
    
    # Check for common schedule patterns
    regular_minutes = [0, 15, 30, 45]
    regular_count = sum(minutes.isin(regular_minutes))
    total_count = len(minutes)
    
    regularity_ratio = regular_count / total_count
    
    # Consider it regular if >50% of timestamps fall on regular minutes
    is_regular = regularity_ratio > 0.5
    
    return is_regular, regularity_ratio
