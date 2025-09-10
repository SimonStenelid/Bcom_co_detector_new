"""Feature engineering module for cohort-level analysis."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from .config import Config
from .utils.time import (
    compute_hourly_histogram, compute_hourly_entropy, compute_gini_coefficient,
    compute_time_gaps, compute_gap_stats, detect_ladder_sweep,
    is_night_time, compute_ewma, compute_jaccard_similarity,
    detect_schedule_regularity
)
from .utils.signatures import (
    compute_signature_stats, compute_route_diversity, compute_top_signatures,
    compute_signature_jaccard, extract_signature_features
)
from .utils.stats import (
    compute_http_version_stats, compute_status_code_stats, compute_error_bursts,
    compute_ua_churn_per_ip, compute_session_stats, compute_hygiene_stats,
    compute_volume_stats
)


def build_features(date: str, config: Config) -> Dict[str, Any]:
    """
    Build cohort-level features from raw data.
    
    Args:
        date: Date string in YYYY-MM-DD format
        config: Configuration object
        
    Returns:
        Dictionary with feature building results
    """
    logger = logging.getLogger(__name__)
    
    # Load raw data
    raw_path = config.get_paths(date)['raw']
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    
    logger.info(f"Loading raw data from: {raw_path}")
    df = pd.read_parquet(raw_path)

    # Compute global IP diversity per full user agent for this date
    ua_full_col = 'user_agent_full' if 'user_agent_full' in df.columns else None
    ua_full_ip_diversity_map = {}
    total_unique_ips = df['ip'].nunique() if 'ip' in df.columns else 0
    if ua_full_col and 'ip' in df.columns:
        ua_full_ip_diversity_map = (
            df.groupby(ua_full_col)['ip'].nunique().to_dict()
        )
    
    # Filter to minimum events per cohort
    cohort_sizes = df['cohort_key'].value_counts()
    valid_cohorts = cohort_sizes[cohort_sizes >= config.ingest.min_events_per_cohort].index
    df_filtered = df[df['cohort_key'].isin(valid_cohorts)]
    
    logger.info(f"Processing {len(valid_cohorts)} cohorts with >= {config.ingest.min_events_per_cohort} events")
    
    # Build features for each cohort
    cohort_features = []
    
    for cohort_key in valid_cohorts:
        cohort_data = df_filtered[df_filtered['cohort_key'] == cohort_key]
        
        try:
            features = _build_cohort_features(
                cohort_data,
                config,
                ua_full_ip_diversity_map=ua_full_ip_diversity_map,
                total_unique_ips=total_unique_ips,
            )
            cohort_features.append(features)
        except Exception as e:
            logger.warning(f"Error building features for cohort {cohort_key}: {e}")
            continue
    
    # Combine all cohort features
    features_df = pd.DataFrame(cohort_features)
    
    # Add historical features if available
    features_df = _add_historical_features(features_df, date, config)
    
    # Save features
    output_path = config.get_paths(date)['features']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving features to: {output_path}")
    features_df.to_parquet(output_path, index=False, compression='snappy')
    
    return {
        'cohorts_processed': len(features_df),
        'output_path': str(output_path),
        'feature_columns': list(features_df.columns)
    }


def _build_cohort_features(
    cohort_data: pd.DataFrame,
    config: Config,
    ua_full_ip_diversity_map: Dict[str, int] | None = None,
    total_unique_ips: int | None = None,
) -> Dict[str, Any]:
    """Build features for a single cohort."""
    
    # Basic cohort info
    features = {
        'cohort_key': cohort_data['cohort_key'].iloc[0],
        'date': cohort_data['date'].iloc[0],
        'site': cohort_data['site'].iloc[0],
        'partner': cohort_data['partner'].iloc[0],
        'ip': cohort_data['ip'].iloc[0],
        'user_agent': cohort_data['user_agent'].iloc[0],
    }
    
    # Add enriched user agent information if available
    if 'user_agent_type' in cohort_data.columns:
        features['user_agent_type'] = cohort_data['user_agent_type'].iloc[0]
    if 'device_type' in cohort_data.columns:
        features['device_type'] = cohort_data['device_type'].iloc[0]
    if 'os_type' in cohort_data.columns:
        features['os_type'] = cohort_data['os_type'].iloc[0]

    # Full user agent concentration within cohort (exact string)
    if 'user_agent_full' in cohort_data.columns:
        ua_full_counts = cohort_data['user_agent_full'].value_counts()
        if len(ua_full_counts) > 0:
            ua_full_mode = ua_full_counts.index[0]
            ua_full_mode_share = ua_full_counts.iloc[0] / len(cohort_data)
            features['ua_full'] = ua_full_mode
            features['ua_full_mode_share'] = float(ua_full_mode_share)
            # IP diversity for this UA full across the full day
            if ua_full_ip_diversity_map is not None and total_unique_ips:
                ip_div = int(ua_full_ip_diversity_map.get(ua_full_mode, 1))
                features['ua_full_ip_diversity'] = ip_div
                features['ua_full_ip_diversity_ratio'] = (
                    float(ip_div) / float(total_unique_ips)
                ) if total_unique_ips > 0 else 0.0
        else:
            features['ua_full'] = 'Unknown'
            features['ua_full_mode_share'] = 0.0
            features['ua_full_ip_diversity'] = 1
            features['ua_full_ip_diversity_ratio'] = 0.0

    # Time-based features
    time_features = _build_time_features(cohort_data, config)
    features.update(time_features)
    
    # Signature-based features
    signature_features = _build_signature_features(cohort_data)
    features.update(signature_features)
    
    # Client/network features
    client_features = _build_client_features(cohort_data)
    features.update(client_features)
    
    # Volume features
    volume_features = _build_volume_features(cohort_data)
    features.update(volume_features)
    
    # Outcome features
    outcome_features = _build_outcome_features(cohort_data)
    features.update(outcome_features)
    
    # Order/conversion features
    order_features = _build_order_features(cohort_data)
    features.update(order_features)
    
    return features


def _build_time_features(cohort_data: pd.DataFrame, config: Config) -> Dict[str, Any]:
    """Build time-based features."""
    features = {}
    
    if 'timestamp' in cohort_data.columns and len(cohort_data) > 0:
        timestamps = cohort_data['timestamp']
        hours = timestamps.dt.hour
        
        # Hourly histogram
        hourly_hist = compute_hourly_histogram(hours)
        for i, prop in enumerate(hourly_hist):
            features[f'hourly_hist_{i:02d}'] = prop
        
        # Time-based statistics
        features['hourly_entropy'] = compute_hourly_entropy(hourly_hist)
        features['hourly_gini'] = compute_gini_coefficient(hourly_hist)
        
        # Night time activity
        night_hours = is_night_time(hours, config.features.night_hours)
        features['night_share'] = night_hours.mean()
        
        # Time gaps
        gaps = compute_time_gaps(timestamps)
        gap_stats = compute_gap_stats(gaps, config.features.gap_percentiles)
        features.update(gap_stats)
        
        # Schedule regularity
        is_regular, regularity_ratio = detect_schedule_regularity(timestamps)
        features['schedule_regular_times'] = is_regular
        features['schedule_regularity_strength'] = regularity_ratio
        
        # First and last timestamps
        features['first_timestamp'] = timestamps.min()
        features['last_timestamp'] = timestamps.max()
        features['duration_hours'] = (timestamps.max() - timestamps.min()).total_seconds() / 3600
        
    else:
        # Default values when no timestamps
        for i in range(24):
            features[f'hourly_hist_{i:02d}'] = 0.0
        features.update({
            'hourly_entropy': 0.0,
            'hourly_gini': 0.0,
            'night_share': 0.0,
            'gap_p50': 0.0,
            'gap_p95': 0.0,
            'gap_p95_over_p50': 0.0,
            'max_run_lt_1s': 0,
            'share_lt_2s': 0.0,
            'schedule_regular_times': False,
            'schedule_regularity_strength': 0.0,
            'duration_hours': 0.0
        })
    
    return features


def _build_signature_features(cohort_data: pd.DataFrame) -> Dict[str, Any]:
    """Build signature-based features."""
    features = {}
    
    if 'payload_signature' in cohort_data.columns:
        signatures = cohort_data['payload_signature']
        
        # Basic signature stats
        sig_stats = compute_signature_stats(signatures)
        features.update(sig_stats)
        
        # Route diversity
        route_diversity = compute_route_diversity(
            cohort_data.get('origin', pd.Series()),
            cohort_data.get('destination', pd.Series()),
            cohort_data.get('dep_date', pd.Series())
        )
        features.update(route_diversity)
        
        # Ladder sweep detection
        if 'dep_date' in cohort_data.columns:
            is_ladder, ladder_count = detect_ladder_sweep(
                cohort_data['dep_date'], 
                max_gap_days=3
            )
            features['ladder_sweep'] = is_ladder
            features['ladder_sweep_count'] = ladder_count
        else:
            features['ladder_sweep'] = False
            features['ladder_sweep_count'] = 0
        
        # Top signatures for evidence
        top_sigs = compute_top_signatures(signatures, top_n=3)
        for i, sig_info in enumerate(top_sigs):
            features[f'top_sig_{i+1}_count'] = sig_info['count']
            features[f'top_sig_{i+1}_pct'] = sig_info['percentage']
    
    else:
        # Default values
        features.update({
            'num_signatures': 0,
            'top_signature_share': 0.0,
            'num_exact_repeats_ge_10': 0,
            'repeat_ratio': 0.0,
            'od_unique': 0,
            'dep_dates_unique': 0,
            'ladder_sweep': False,
            'ladder_sweep_count': 0
        })
    
    return features


def _build_client_features(cohort_data: pd.DataFrame) -> Dict[str, Any]:
    """Build client and network features."""
    features = {}
    
    # HTTP version stats
    if 'http_version' in cohort_data.columns:
        http_stats = compute_http_version_stats(cohort_data['http_version'])
        features.update(http_stats)
    else:
        features.update({
            'http11_share': 0.0,
            'http2_share': 0.0,
            'http3_share': 0.0,
            'http_unknown_share': 0.0
        })
    
    # Hygiene stats
    hygiene_stats = compute_hygiene_stats(
        cohort_data.get('cookie_present', pd.Series()),
        cohort_data.get('referrer', pd.Series())
    )
    features.update(hygiene_stats)
    
    # User agent churn per IP
    if 'ip' in cohort_data.columns and 'user_agent' in cohort_data.columns:
        ua_churn = compute_ua_churn_per_ip(cohort_data['ip'], cohort_data['user_agent'])
        features['ua_churn_per_ip_day'] = ua_churn.iloc[0] if len(ua_churn) > 0 else 1
    else:
        features['ua_churn_per_ip_day'] = 1
    
    # User agent diversity analysis
    if 'user_agent_type' in cohort_data.columns:
        ua_types = cohort_data['user_agent_type'].value_counts()
        features['ua_type_diversity'] = len(ua_types)
        features['is_bot_ua_type'] = (cohort_data['user_agent_type'] == 'Bot').any()
        features['is_application_ua_type'] = (cohort_data['user_agent_type'] == 'Application').any()
    else:
        features['ua_type_diversity'] = 1
        features['is_bot_ua_type'] = False
        features['is_application_ua_type'] = False
    
    # Device type analysis
    if 'device_type' in cohort_data.columns:
        device_types = cohort_data['device_type'].value_counts()
        features['device_diversity'] = len(device_types)
        features['is_mobile_device'] = cohort_data['device_type'].str.contains('Phone|Tablet', case=False, na=False).any()
    else:
        features['device_diversity'] = 1
        features['is_mobile_device'] = False
    
    return features


def _build_volume_features(cohort_data: pd.DataFrame) -> Dict[str, Any]:
    """Build volume-related features."""
    features = {}
    
    n_events = len(cohort_data)
    
    # Session stats
    session_stats = compute_session_stats(
        cohort_data.get('session_id', pd.Series()),
        n_events
    )
    features.update(session_stats)
    
    # Volume stats
    volume_stats = compute_volume_stats(n_events, session_stats['unique_sessions'])
    features.update(volume_stats)
    
    return features


def _build_outcome_features(cohort_data: pd.DataFrame) -> Dict[str, Any]:
    """Build outcome-related features."""
    features = {}
    
    # Status code stats
    if 'status_code' in cohort_data.columns:
        status_stats = compute_status_code_stats(cohort_data['status_code'])
        features.update(status_stats)
        
        # Error bursts
        features['error_bursts'] = compute_error_bursts(cohort_data['status_code'])
    else:
        features.update({
            'ok2xx_rate': 1.0,
            'client4xx_rate': 0.0,
            'server5xx_rate': 0.0,
            'error_bursts': 0
        })
    
    return features


def _build_order_features(cohort_data: pd.DataFrame) -> Dict[str, Any]:
    """Build order/conversion-related features."""
    features = {}
    
    n_events = len(cohort_data)
    
    # Order statistics
    if 'num_orders' in cohort_data.columns:
        total_orders = cohort_data['num_orders'].sum()
        orders_per_event = total_orders / n_events if n_events > 0 else 0
        
        # Conversion rate
        conversion_rate = (cohort_data['num_orders'] > 0).mean() if n_events > 0 else 0
        
        # Order distribution
        events_with_orders = (cohort_data['num_orders'] > 0).sum()
        events_without_orders = n_events - events_with_orders
        
        features.update({
            'total_orders': total_orders,
            'orders_per_event': orders_per_event,
            'conversion_rate': conversion_rate,
            'events_with_orders': events_with_orders,
            'events_without_orders': events_without_orders,
            'order_ratio': events_with_orders / n_events if n_events > 0 else 0
        })
        
        # Order patterns (potential bot indicators)
        if total_orders > 0:
            # Check for suspicious order patterns
            order_values = cohort_data['num_orders'].values
            unique_order_values = len(set(order_values))
            max_orders = order_values.max()
            min_orders = order_values.min()
            
            # Suspicious patterns
            all_same_orders = unique_order_values == 1  # All events have same number of orders
            only_zero_or_one = set(order_values).issubset({0, 1})  # Only 0 or 1 orders
            
            features.update({
                'unique_order_values': unique_order_values,
                'max_orders_per_event': max_orders,
                'min_orders_per_event': min_orders,
                'all_same_orders': all_same_orders,
                'only_zero_or_one_orders': only_zero_or_one,
                'order_variance': order_values.var() if len(order_values) > 1 else 0
            })
        else:
            features.update({
                'unique_order_values': 0,
                'max_orders_per_event': 0,
                'min_orders_per_event': 0,
                'all_same_orders': True,
                'only_zero_or_one_orders': True,
                'order_variance': 0
            })
    else:
        # Default values if no order data
        features.update({
            'total_orders': 0,
            'orders_per_event': 0,
            'conversion_rate': 0,
            'events_with_orders': 0,
            'events_without_orders': n_events,
            'order_ratio': 0,
            'unique_order_values': 0,
            'max_orders_per_event': 0,
            'min_orders_per_event': 0,
            'all_same_orders': True,
            'only_zero_or_one_orders': True,
            'order_variance': 0
        })
    
    return features


def _add_historical_features(features_df: pd.DataFrame, date: str, config: Config) -> pd.DataFrame:
    """Add historical features using EWMA and cross-day comparisons."""
    logger = logging.getLogger(__name__)
    
    # Try to load historical data
    historical_features = _load_historical_features(date, config)
    
    if historical_features is None or len(historical_features) == 0:
        logger.info("No historical data available, skipping historical features")
        return features_df
    
    # Add EWMA features
    ewma_features = _compute_ewma_features(features_df, historical_features, config)
    features_df = pd.concat([features_df, ewma_features], axis=1)
    
    # Add cross-day comparison features
    comparison_features = _compute_comparison_features(features_df, historical_features)
    features_df = pd.concat([features_df, comparison_features], axis=1)
    
    logger.info(f"Added {len(ewma_features.columns) + len(comparison_features.columns)} historical features")
    
    return features_df


def _load_historical_features(date: str, config: Config) -> Optional[pd.DataFrame]:
    """Load historical features for the past N days."""
    logger = logging.getLogger(__name__)
    
    current_date = datetime.strptime(date, '%Y-%m-%d')
    historical_data = []
    
    for i in range(1, config.history.lookback_days + 1):
        hist_date = current_date - timedelta(days=i)
        hist_date_str = hist_date.strftime('%Y-%m-%d')
        
        hist_path = config.get_paths(hist_date_str)['features']
        if hist_path.exists():
            try:
                hist_df = pd.read_parquet(hist_path)
                hist_df['hist_date'] = hist_date_str
                historical_data.append(hist_df)
            except Exception as e:
                logger.warning(f"Error loading historical data for {hist_date_str}: {e}")
    
    if not historical_data:
        return None
    
    return pd.concat(historical_data, ignore_index=True)


def _compute_ewma_features(
    current_features: pd.DataFrame, 
    historical_features: pd.DataFrame, 
    config: Config
) -> pd.DataFrame:
    """Compute EWMA features from historical data."""
    
    ewma_features = pd.DataFrame(index=current_features.index)
    
    # Features to compute EWMA for
    ewma_columns = [
        'night_share', 'top_signature_share', 'gap_p50', 'gap_p95',
        'hourly_entropy', 'repeat_ratio', 'cookie_rate', 'referrer_rate'
    ]
    
    for col in ewma_columns:
        if col in current_features.columns:
            # Get historical values for this cohort
            cohort_key = current_features['cohort_key']
            
            for idx, cohort in cohort_key.items():
                hist_data = historical_features[
                    historical_features['cohort_key'] == cohort
                ][col].dropna()
                
                if len(hist_data) > 0:
                    # Compute EWMA
                    ewma_7d = compute_ewma(hist_data, config.history.ewma_alpha_7d).iloc[-1]
                    ewma_30d = compute_ewma(hist_data, config.history.ewma_alpha_30d).iloc[-1]
                    
                    ewma_features.loc[idx, f'ewma_{col}_7d'] = ewma_7d
                    ewma_features.loc[idx, f'ewma_{col}_30d'] = ewma_30d
                else:
                    ewma_features.loc[idx, f'ewma_{col}_7d'] = current_features.loc[idx, col]
                    ewma_features.loc[idx, f'ewma_{col}_30d'] = current_features.loc[idx, col]
    
    return ewma_features.fillna(0.0)


def _compute_comparison_features(
    current_features: pd.DataFrame, 
    historical_features: pd.DataFrame
) -> pd.DataFrame:
    """Compute cross-day comparison features."""
    
    comparison_features = pd.DataFrame(index=current_features.index)
    
    # Get yesterday's data
    yesterday_data = historical_features[
        historical_features['hist_date'] == historical_features['hist_date'].max()
    ]
    
    if len(yesterday_data) == 0:
        return comparison_features
    
    # Signature Jaccard similarity
    for idx, row in current_features.iterrows():
        cohort_key = row['cohort_key']
        
        # Get current and yesterday's signatures
        current_sigs = set()  # Would need to be computed from raw data
        yesterday_sigs = set()  # Would need to be computed from raw data
        
        jaccard_sim = compute_jaccard_similarity(current_sigs, yesterday_sigs)
        comparison_features.loc[idx, 'jaccard_signatures_vs_prev_day'] = jaccard_sim
    
    return comparison_features.fillna(0.0)
