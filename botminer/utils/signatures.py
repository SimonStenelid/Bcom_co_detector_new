"""Signature computation utilities for detecting repetitive patterns."""

import hashlib
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set


def compute_payload_signature(
    origin: str,
    destination: str, 
    dep_date: str,
    ret_date: str,
    pax_config: str,
    cabin: str,
    stops: str
) -> str:
    """Compute SHA1 hash of normalized payload signature."""
    # Normalize and concatenate fields
    normalized = "|".join([
        str(origin or "").lower().strip(),
        str(destination or "").lower().strip(),
        str(dep_date or "").lower().strip(),
        str(ret_date or "").lower().strip(),
        str(pax_config or "").lower().strip(),
        str(cabin or "").lower().strip(),
        str(stops or "").lower().strip(),
    ])
    
    # Compute SHA1 hash
    return hashlib.sha1(normalized.encode('utf-8')).hexdigest()


def compute_signature_stats(signatures: pd.Series) -> Dict[str, float]:
    """Compute statistics about signature repetition."""
    if len(signatures) == 0:
        return {
            "num_signatures": 0,
            "top_signature_share": 0.0,
            "num_exact_repeats_ge_10": 0,
            "repeat_ratio": 0.0
        }
    
    # Count signature frequencies
    sig_counts = signatures.value_counts()
    total_events = len(signatures)
    
    # Top signature share
    top_signature_share = sig_counts.iloc[0] / total_events if len(sig_counts) > 0 else 0.0
    
    # Number of unique signatures
    num_signatures = len(sig_counts)
    
    # Count signatures that appear 10+ times
    num_exact_repeats_ge_10 = (sig_counts >= 10).sum()
    
    # Repeat ratio (exact repeats / total events)
    repeat_ratio = (sig_counts - 1).sum() / total_events if total_events > 0 else 0.0
    
    return {
        "num_signatures": num_signatures,
        "top_signature_share": top_signature_share,
        "num_exact_repeats_ge_10": num_exact_repeats_ge_10,
        "repeat_ratio": repeat_ratio
    }


def compute_route_diversity(
    origins: pd.Series,
    destinations: pd.Series,
    dep_dates: pd.Series
) -> Dict[str, int]:
    """Compute route and date diversity metrics."""
    # Origin-destination pairs
    od_pairs = set(zip(origins.fillna(""), destinations.fillna("")))
    od_unique = len(od_pairs)
    
    # Unique departure dates
    dep_dates_unique = dep_dates.nunique()
    
    return {
        "od_unique": od_unique,
        "dep_dates_unique": dep_dates_unique
    }


def compute_top_signatures(signatures: pd.Series, top_n: int = 3) -> List[Dict]:
    """Get top N signatures with their counts and percentages."""
    if len(signatures) == 0:
        return []
    
    sig_counts = signatures.value_counts()
    total = len(signatures)
    
    top_signatures = []
    for i, (sig, count) in enumerate(sig_counts.head(top_n).items()):
        top_signatures.append({
            "signature": sig,
            "count": count,
            "percentage": count / total * 100
        })
    
    return top_signatures


def compute_signature_jaccard(
    signatures_today: Set[str], 
    signatures_yesterday: Set[str]
) -> float:
    """Compute Jaccard similarity between signature sets from two days."""
    if not signatures_today and not signatures_yesterday:
        return 1.0
    if not signatures_today or not signatures_yesterday:
        return 0.0
    
    intersection = len(signatures_today.intersection(signatures_yesterday))
    union = len(signatures_today.union(signatures_yesterday))
    
    return intersection / union if union > 0 else 0.0


def detect_pattern_anomalies(signatures: pd.Series) -> Dict[str, bool]:
    """Detect various pattern anomalies in signatures."""
    if len(signatures) < 10:
        return {
            "high_repetition": False,
            "low_diversity": False,
            "burst_pattern": False
        }
    
    sig_counts = signatures.value_counts()
    total = len(signatures)
    
    # High repetition: top signature > 50% of events
    high_repetition = sig_counts.iloc[0] / total > 0.5 if len(sig_counts) > 0 else False
    
    # Low diversity: < 5 unique signatures for 100+ events
    low_diversity = len(sig_counts) < 5 and total >= 100
    
    # Burst pattern: check for temporal clustering
    # This is a simplified check - in practice you'd analyze timestamps
    burst_pattern = False
    
    return {
        "high_repetition": high_repetition,
        "low_diversity": low_diversity,
        "burst_pattern": burst_pattern
    }


def extract_signature_features(
    signatures: pd.Series,
    origins: pd.Series,
    destinations: pd.Series,
    dep_dates: pd.Series
) -> Dict[str, float]:
    """Extract comprehensive signature-based features."""
    # Basic signature stats
    sig_stats = compute_signature_stats(signatures)
    
    # Route diversity
    route_diversity = compute_route_diversity(origins, destinations, dep_dates)
    
    # Pattern anomalies
    anomalies = detect_pattern_anomalies(signatures)
    
    # Combine all features
    features = {
        **sig_stats,
        **route_diversity,
        **{f"anomaly_{k}": float(v) for k, v in anomalies.items()}
    }
    
    return features
