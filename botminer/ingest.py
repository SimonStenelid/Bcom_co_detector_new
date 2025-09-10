"""Data ingestion module for processing CSV files."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from .config import Config
from .utils.time import parse_timestamp
from .utils.user_agents import enrich_user_agent, get_browser_type, get_user_agent_type, get_device_type, get_os_type


def ingest_csv(
    csv_path: str, 
    date: str, 
    config: Config,
    ip_csv_path: str = None
) -> Dict[str, Any]:
    """
    Ingest CSV file and save as Parquet.
    
    Args:
        csv_path: Path to input CSV file
        date: Date string in YYYY-MM-DD format
        config: Configuration object
        ip_csv_path: Optional path to IP address CSV file for merging
        
    Returns:
        Dictionary with ingestion results
    """
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Create output directory
    output_path = config.get_paths(date)['raw']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Reading CSV file: {csv_path}")
    
    try:
        # Read main CSV with robust error handling
        df = pd.read_csv(
            csv_path,
            encoding='utf-8',
            low_memory=False,
            na_values=['', 'null', 'NULL', 'None', 'N/A', 'n/a']
        )
        
        logger.info(f"Loaded {len(df)} rows from main CSV")
        
        # Read and merge IP data if provided
        if ip_csv_path and Path(ip_csv_path).exists():
            logger.info(f"Loading IP data from: {ip_csv_path}")
            ip_df = pd.read_csv(
                ip_csv_path,
                encoding='utf-8',
                low_memory=False,
                na_values=['', 'null', 'NULL', 'None', 'N/A', 'n/a']
            )
            logger.info(f"Loaded {len(ip_df)} rows from IP CSV")
            
            # Merge the datasets
            df = _merge_ip_data(df, ip_df)
        else:
            logger.warning("No IP CSV provided or file not found, using default IP values")
            df['ip'] = 'unknown'
        
        # Map columns to expected format
        df = _map_columns(df)
        
        # Process and normalize data
        df = _normalize_dataframe(df, config)
        
        # Add derived columns
        df = _add_derived_columns(df, date, config)
        
        # Save as Parquet
        logger.info(f"Saving to Parquet: {output_path}")
        df.to_parquet(output_path, index=False, compression='snappy')
        
        return {
            'rows_processed': len(df),
            'output_path': str(output_path),
            'columns': list(df.columns),
            'date': date
        }
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise


def _merge_ip_data(df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """Merge IP data with main dataframe using the ID field."""
    logger = logging.getLogger(__name__)
    
    # Clean column names in IP dataframe - handle the escaped dots properly (mdc\.addr -> addr)
    ip_df.columns = ip_df.columns.str.replace(r'mdc\.', '', regex=True)
    
    logger.info(f"IP dataframe columns after cleaning: {list(ip_df.columns)}")
    
    # The click ID field in the main df should match the 'click' field in IP df
    # New booking exports have two 'ID' columns: the first is the 4-digit user agent ID,
    # and the second (last) ID column is the click ID. Select the last ID-like column.
    id_candidates = [col for col in df.columns if str(col).strip().upper().startswith('ID')]
    if not id_candidates:
        # Fallback to previous heuristic but log a warning
        logger.warning("No explicit 'ID' columns found; defaulting to last column for click ID merge")
        id_col = df.columns[-1]
    else:
        id_col = id_candidates[-1]
    logger.info(f"Using click ID column for merge: {id_col}")
    
    # Check if the required columns exist
    if 'click' not in ip_df.columns or 'addr' not in ip_df.columns:
        logger.error(f"Required columns not found in IP data. Available: {list(ip_df.columns)}")
        # Try alternative column names
        click_col = None
        addr_col = None
        for col in ip_df.columns:
            if 'click' in col.lower():
                click_col = col
            if 'addr' in col.lower():
                addr_col = col
        
        if click_col and addr_col:
            logger.info(f"Using alternative columns: {click_col}, {addr_col}")
            ip_df = ip_df.rename(columns={click_col: 'click', addr_col: 'addr'})
        else:
            raise ValueError(f"Cannot find click/addr columns in IP data: {list(ip_df.columns)}")
    
    # Convert ID columns to string to ensure consistent data types
    df[id_col] = df[id_col].astype(str)
    ip_df['click'] = ip_df['click'].astype(str)
    
    logger.info(f"Sample ID values from main df: {df[id_col].head().tolist()}")
    logger.info(f"Sample click values from IP df: {ip_df['click'].head().tolist()}")
    
    # Merge on ID = click
    merged_df = df.merge(
        ip_df[['click', 'addr']], 
        left_on=id_col, 
        right_on='click', 
        how='left'
    )
    
    # Add IP address to the dataframe
    merged_df['ip'] = merged_df['addr'].fillna('unknown')
    
    # Drop the temporary columns
    merged_df = merged_df.drop(['click', 'addr'], axis=1, errors='ignore')
    
    logger.info(f"Merged datasets: {len(merged_df)} rows, {merged_df['ip'].notna().sum()} with IP addresses")
    
    return merged_df


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map the actual column names to expected format."""
    logger = logging.getLogger(__name__)
    
    # Column mapping from actual CSV to expected format
    column_mapping = {
        'Created Time': 'timestamp',
        'Site': 'site', 
        'Partner Domain': 'partner',
        'OriginCity': 'origin',
        'DestinationCity': 'destination',
        'JourneyType': 'journey_type',
        'Out Date': 'dep_date',
        'Return Date': 'ret_date',
        'Num Adt': 'num_adults',
        'Num Cnn': 'num_children', 
        'Num Inf': 'num_infants',
        'Cabin Classes': 'cabin',
        'Total Stops': 'stops',
        'MarketingCarriers': 'carrier',
        'Sessions ID': 'session_id',
        'ID': 'user_agent_id'  # This is the 4-letter ID (user agent)
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Create pax_config from individual passenger counts
    if 'num_adults' in df.columns and 'num_children' in df.columns and 'num_infants' in df.columns:
        df['pax_config'] = (
            df['num_adults'].astype(str) + '-' + 
            df['num_children'].astype(str) + '-' + 
            df['num_infants'].astype(str)
        )
    else:
        df['pax_config'] = '1-0-0'  # Default
    
    # Enrich user agent information
    if 'user_agent_id' in df.columns:
        logger.info("Enriching user agent information...")
        
        # Add enriched user agent columns
        df['user_agent'] = df['user_agent_id'].apply(
            lambda x: get_browser_type(int(x)) if pd.notna(x) else 'Unknown'
        )
        df['user_agent_type'] = df['user_agent_id'].apply(
            lambda x: get_user_agent_type(int(x)) if pd.notna(x) else 'Unknown'
        )
        df['device_type'] = df['user_agent_id'].apply(
            lambda x: get_device_type(int(x)) if pd.notna(x) else 'Unknown'
        )
        df['os_type'] = df['user_agent_id'].apply(
            lambda x: get_os_type(int(x)) if pd.notna(x) else 'Unknown'
        )
        
        # Get full user agent name for detailed analysis
        df['user_agent_full'] = df['user_agent_id'].apply(
            lambda x: enrich_user_agent(int(x))['name'] if pd.notna(x) else 'Unknown'
        )
    else:
        logger.warning("No user_agent_id column found, using default values")
        df['user_agent'] = 'unknown'
        df['user_agent_type'] = 'Unknown'
        df['device_type'] = 'Unknown'
        df['os_type'] = 'Unknown'
        df['user_agent_full'] = 'Unknown'
    
    # Set default values for missing columns
    defaults = {
        'referrer': None,
        'cookie_present': False,
        'http_version': 'unknown',
        'status_code': 200,
        'request_id': None
    }
    
    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value
    
    logger.info(f"Mapped columns: {list(df.columns)}")
    
    return df


def _normalize_dataframe(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Normalize and clean the dataframe."""
    logger = logging.getLogger(__name__)
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Normalize timestamp
    if 'timestamp' in df.columns:
        logger.info("Normalizing timestamps...")
        df['timestamp'] = df['timestamp'].apply(
            lambda x: parse_timestamp(str(x), config.ingest.tz)
        )
    
    # Normalize string columns
    string_columns = [
        'site', 'partner', 'origin', 'destination', 'journey_type',
        'dep_date', 'ret_date', 'pax_config', 'cabin', 'stops',
        'ip', 'user_agent', 'session_id', 'referrer', 'http_version'
    ]
    
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', None)
            df[col] = df[col].replace('None', None)
    
    # Normalize boolean columns
    if 'cookie_present' in df.columns:
        df['cookie_present'] = df['cookie_present'].astype(bool)
    
    # Normalize numeric columns
    numeric_columns = ['status_code']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values
    df = _handle_missing_values(df)
    
    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with appropriate defaults."""
    logger = logging.getLogger(__name__)
    
    # Fill missing values with appropriate defaults
    defaults = {
        'site': 'unknown',
        'partner': 'unknown',
        'origin': 'unknown',
        'destination': 'unknown',
        'journey_type': 'OW',
        'dep_date': None,
        'ret_date': None,
        'pax_config': '1-0-0',
        'cabin': 'economy',
        'stops': '0',
        'ip': 'unknown',
        'user_agent': 'unknown',
        'session_id': None,
        'referrer': None,
        'cookie_present': False,
        'http_version': 'unknown',
        'status_code': 200,
        'request_id': None
    }
    
    for col, default_value in defaults.items():
        if col in df.columns:
            if df[col].isna().sum() > 0:
                logger.info(f"Filling {df[col].isna().sum()} missing values in {col}")
                if default_value is None:
                    # Don't fill None values, leave them as is
                    continue
                df[col] = df[col].fillna(default_value)
    
    return df


def _add_derived_columns(df: pd.DataFrame, date: str, config: Config) -> pd.DataFrame:
    """Add derived columns for analysis."""
    logger = logging.getLogger(__name__)
    
    # Add date column
    df['date'] = date
    
    # Add cohort key components
    df['cohort_date'] = date
    df['cohort_site'] = df['site'].fillna('unknown')
    df['cohort_partner'] = df['partner'].fillna('unknown')
    df['cohort_ip'] = df['ip'].fillna('unknown')
    df['cohort_ua'] = df['user_agent'].fillna('unknown')
    
    # Create cohort key
    df['cohort_key'] = (
        df['cohort_date'].astype(str) + '|' +
        df['cohort_site'].astype(str) + '|' +
        df['cohort_partner'].astype(str) + '|' +
        df['cohort_ip'].astype(str) + '|' +
        df['cohort_ua'].astype(str)
    )
    
    # Add hour from timestamp
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    else:
        df['hour'] = 0
        df['minute'] = 0
        df['day_of_week'] = 0
    
    # Add payload signature
    df['payload_signature'] = _compute_payload_signatures(df)
    
    logger.info(f"Added derived columns. Final shape: {df.shape}")
    
    return df


def _compute_payload_signatures(df: pd.DataFrame) -> pd.Series:
    """Compute payload signatures for each row."""
    from .utils.signatures import compute_payload_signature
    
    def compute_sig(row):
        return compute_payload_signature(
            origin=row.get('origin', ''),
            destination=row.get('destination', ''),
            dep_date=row.get('dep_date', ''),
            ret_date=row.get('ret_date', ''),
            pax_config=row.get('pax_config', ''),
            cabin=row.get('cabin', ''),
            stops=row.get('stops', '')
        )
    
    return df.apply(compute_sig, axis=1)


def validate_ingestion(df: pd.DataFrame, config: Config) -> Dict[str, Any]:
    """Validate ingested data quality."""
    logger = logging.getLogger(__name__)
    
    validation_results = {
        'total_rows': len(df),
        'cohorts': df['cohort_key'].nunique() if 'cohort_key' in df.columns else 0,
        'date_range': None,
        'missing_data': {},
        'data_quality_issues': []
    }
    
    # Check date range
    if 'timestamp' in df.columns:
        validation_results['date_range'] = {
            'min': df['timestamp'].min(),
            'max': df['timestamp'].max()
        }
    
    # Check missing data
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            validation_results['missing_data'][col] = missing_count
    
    # Check for data quality issues
    if validation_results['total_rows'] == 0:
        validation_results['data_quality_issues'].append("No data rows found")
    
    if validation_results['cohorts'] == 0:
        validation_results['data_quality_issues'].append("No cohorts found")
    
    # Check for minimum events per cohort
    if 'cohort_key' in df.columns:
        cohort_sizes = df['cohort_key'].value_counts()
        small_cohorts = (cohort_sizes < config.ingest.min_events_per_cohort).sum()
        if small_cohorts > 0:
            validation_results['data_quality_issues'].append(
                f"{small_cohorts} cohorts have < {config.ingest.min_events_per_cohort} events"
            )
    
    logger.info(f"Validation results: {validation_results}")
    
    return validation_results
