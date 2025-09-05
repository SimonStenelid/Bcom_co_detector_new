"""Batch ingestion module for processing all data files in the data folder."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime
import glob

from .config import Config
from .utils.time import parse_timestamp
from .utils.user_agents import enrich_user_agent, get_browser_type, get_user_agent_type, get_device_type, get_os_type

logger = logging.getLogger(__name__)


def ingest_all_data(
    data_dir: str = "data/raw",
    date: str = None,
    config: Config = None
) -> Dict[str, Any]:
    """
    Ingest all data files in the data directory and merge them.
    
    Args:
        data_dir: Directory containing the data files
        date: Date string in YYYY-MM-DD format (if None, uses today)
        config: Configuration object
        
    Returns:
        Dictionary with ingestion results
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    if config is None:
        config = Config.from_file("config.yaml")
    
    logger.info(f"Starting batch ingestion for date: {date}")
    logger.info(f"Data directory: {data_dir}")
    
    # Find all data files
    data_path = Path(data_dir)
    booking_files = list(data_path.glob("booking_*.csv"))
    ip_files = list(data_path.glob("On_demand_report_*.csv"))
    
    # Enhanced progress logging
    total_booking_size = sum(f.stat().st_size for f in booking_files) / (1024*1024)  # MB
    total_ip_size = sum(f.stat().st_size for f in ip_files) / (1024*1024)  # MB
    
    logger.info(f"📂 Data Discovery Summary:")
    logger.info(f"   📋 Booking files: {len(booking_files)} ({total_booking_size:.1f} MB)")
    logger.info(f"   🌐 IP files: {len(ip_files)} ({total_ip_size:.1f} MB)")
    logger.info(f"   📊 Total data volume: {total_booking_size + total_ip_size:.1f} MB")
    
    # List all files being processed
    logger.info(f"📋 Booking files to process:")
    for i, booking_file in enumerate(booking_files, 1):
        file_size = booking_file.stat().st_size / (1024*1024)  # MB
        logger.info(f"   {i}. {booking_file.name} ({file_size:.1f} MB)")
    
    logger.info(f"🌐 IP files to process:")
    for i, ip_file in enumerate(ip_files, 1):
        file_size = ip_file.stat().st_size / (1024*1024)  # MB
        logger.info(f"   {i}. {ip_file.name} ({file_size:.1f} MB)")
    
    if not booking_files:
        raise ValueError("No booking files found in data directory")
    
    if not ip_files:
        raise ValueError("No IP files found in data directory")
    
    # Process all booking files with enhanced progress tracking
    logger.info(f"\n🔄 Step 1/5: Processing booking files...")
    all_booking_data = []
    total_booking_rows = 0
    
    for i, booking_file in enumerate(booking_files, 1):
        logger.info(f"   📋 [{i}/{len(booking_files)}] Processing: {booking_file.name}")
        df = pd.read_csv(booking_file, encoding='utf-8', low_memory=False)
        rows_loaded = len(df)
        total_booking_rows += rows_loaded
        logger.info(f"       ✓ Loaded {rows_loaded:,} rows ({df.shape[1]} columns)")
        all_booking_data.append(df)
    
    # Combine all booking data
    logger.info(f"\n🔄 Step 2/5: Combining booking data...")
    if len(all_booking_data) > 1:
        combined_booking = pd.concat(all_booking_data, ignore_index=True)
        logger.info(f"   ✓ Combined {len(booking_files)} files into {len(combined_booking):,} total rows")
    else:
        combined_booking = all_booking_data[0]
        logger.info(f"   ✓ Using single booking file with {len(combined_booking):,} rows")
    
    # Process all IP files with enhanced progress tracking
    logger.info(f"\n🔄 Step 3/5: Processing IP files...")
    all_ip_data = []
    total_ip_rows = 0
    
    for i, ip_file in enumerate(ip_files, 1):
        logger.info(f"   🌐 [{i}/{len(ip_files)}] Processing: {ip_file.name}")
        df = pd.read_csv(ip_file, encoding='utf-8', low_memory=False)
        rows_loaded = len(df)
        total_ip_rows += rows_loaded
        logger.info(f"       ✓ Loaded {rows_loaded:,} rows ({df.shape[1]} columns)")
        all_ip_data.append(df)
    
    # Combine all IP data
    logger.info(f"\n🔄 Step 4/5: Combining IP data...")
    if len(all_ip_data) > 1:
        combined_ip = pd.concat(all_ip_data, ignore_index=True)
        logger.info(f"   ✓ Combined {len(ip_files)} files into {len(combined_ip):,} total rows")
    else:
        combined_ip = all_ip_data[0]
        logger.info(f"   ✓ Using single IP file with {len(combined_ip):,} rows")
    
    # Data volume summary before merging
    logger.info(f"\n📊 Pre-merge Data Volume Summary:")
    logger.info(f"   📋 Total booking rows: {len(combined_booking):,}")
    logger.info(f"   🌐 Total IP rows: {len(combined_ip):,}")
    logger.info(f"   📈 Combined raw data: {len(combined_booking) + len(combined_ip):,} rows")
    
    # Merge the datasets
    logger.info(f"\n🔄 Step 5/5: Merging datasets...")
    merged_df = _merge_all_data(combined_booking, combined_ip)
    
    # Map columns to expected format
    merged_df = _map_columns_enhanced(merged_df)
    
    # Process and normalize data
    merged_df = _normalize_dataframe(merged_df, config)
    
    # Add derived columns
    merged_df = _add_derived_columns(merged_df, date, config)
    
    # Save as Parquet
    output_path = config.get_paths(date)['raw']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n💾 Saving processed data...")
    logger.info(f"   📁 Output path: {output_path}")
    merged_df.to_parquet(output_path, index=False, compression='snappy')
    
    # Calculate final statistics
    file_size = output_path.stat().st_size / (1024*1024)  # MB
    unique_cohorts = merged_df['cohort_key'].nunique() if 'cohort_key' in merged_df.columns else 0
    unique_ips = merged_df['ip'].nunique() if 'ip' in merged_df.columns else 0
    unique_user_agents = merged_df['user_agent'].nunique() if 'user_agent' in merged_df.columns else 0
    
    logger.info(f"\n🎉 Batch Processing Complete!")
    logger.info(f"   ✅ Final dataset: {len(merged_df):,} rows × {merged_df.shape[1]} columns")
    logger.info(f"   💾 File size: {file_size:.1f} MB (compressed)")
    logger.info(f"   🎯 Unique cohorts: {unique_cohorts:,}")
    logger.info(f"   🌐 Unique IPs: {unique_ips:,}")
    logger.info(f"   🔧 Unique user agents: {unique_user_agents:,}")
    logger.info(f"   📊 Data reduction: {len(combined_booking):,} → {len(merged_df):,} rows ({len(merged_df)/len(combined_booking)*100:.1f}% retained)")
    
    return {
        'rows_processed': len(merged_df),
        'output_path': str(output_path),
        'booking_files': [f.name for f in booking_files],
        'ip_files': [f.name for f in ip_files],
        'merged_rows': len(merged_df),
        'total_input_rows': len(combined_booking) + len(combined_ip),
        'data_reduction_ratio': len(merged_df) / len(combined_booking),
        'file_size_mb': file_size,
        'unique_cohorts': unique_cohorts,
        'unique_ips': unique_ips,
        'unique_user_agents': unique_user_agents
    }


def _merge_all_data(booking_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """Merge all booking and IP data."""
    logger = logging.getLogger(__name__)
    
    # Clean column names in IP dataframe
    ip_df.columns = ip_df.columns.str.replace(r'mdc\\.', '', regex=True)
    logger.info(f"IP dataframe columns after cleaning: {list(ip_df.columns)}")
    
    # Find the ID column in booking df (it's the last column before any additional columns)
    id_columns = [col for col in booking_df.columns if col.startswith('ID')]
    if not id_columns:
        raise ValueError("No ID column found in booking data")
    
    # Use the last ID column (should be the 10-digit click ID)
    id_col = id_columns[-1]
    logger.info(f"Using ID column: {id_col}")
    
    # Check if the required columns exist in IP data
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
    booking_df[id_col] = booking_df[id_col].astype(str)
    ip_df['click'] = ip_df['click'].astype(str)
    
    logger.info(f"Sample ID values from booking df: {booking_df[id_col].head().tolist()}")
    logger.info(f"Sample click values from IP df: {ip_df['click'].head().tolist()}")
    
    # Merge on ID = click (inner join to only keep rows with matching IPs)
    merged_df = booking_df.merge(
        ip_df[['click', 'addr']], 
        left_on=id_col, 
        right_on='click', 
        how='inner'  # Only keep rows where IP can be matched
    )
    
    # Add IP address to the dataframe
    merged_df['ip'] = merged_df['addr']
    
    # Drop the temporary columns
    merged_df = merged_df.drop(['click', 'addr'], axis=1, errors='ignore')
    
    logger.info(f"Merged datasets: {len(merged_df)} rows with IP addresses (from {len(booking_df)} booking rows)")
    logger.info(f"IP match rate: {len(merged_df)/len(booking_df)*100:.1f}%")
    
    return merged_df


def _map_columns_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """Map the actual column names to expected format with enhanced features."""
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
        'ID': 'user_agent_id',  # This is the 4-letter ID (user agent)
        'Num Orders': 'num_orders'  # New order column
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
    
    # Normalize timestamps
    logger.info("Normalizing timestamps...")
    df['timestamp'] = df['timestamp'].apply(parse_timestamp)
    
    # Fill missing values with sensible defaults
    defaults = {
        'ret_date': None,  # One-way trips don't have return dates
        'stops': 0,  # Default to non-stop
        'referrer': None,
        'request_id': None,
        'num_orders': 0  # Default to no orders
    }
    
    for col, default_value in defaults.items():
        if col in df.columns:
            if df[col].isna().sum() > 0:
                logger.info(f"Filling {df[col].isna().sum()} missing values in {col}")
                if default_value is None:
                    # Don't fill None values, leave them as is
                    continue
                df[col] = df[col].fillna(default_value)
    
    # Convert boolean columns
    if 'cookie_present' in df.columns:
        df['cookie_present'] = df['cookie_present'].astype(bool)
    
    # Convert numeric columns
    numeric_columns = ['num_adults', 'num_children', 'num_infants', 'stops', 'status_code', 'num_orders']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def _add_derived_columns(df: pd.DataFrame, date: str, config: Config) -> pd.DataFrame:
    """Add derived columns for analysis."""
    logger = logging.getLogger(__name__)
    
    # Add date column
    df['date'] = date
    
    # Create cohort key
    df['cohort_key'] = (
        df['date'].astype(str) + '|' +
        df['site'].astype(str) + '|' +
        df['partner'].astype(str) + '|' +
        df['ip'].astype(str) + '|' +
        df['user_agent'].astype(str)
    )
    
    # Add order conversion flag
    if 'num_orders' in df.columns:
        df['has_order'] = (df['num_orders'] > 0).astype(int)
        df['order_value'] = df['num_orders']  # Could be enhanced with actual order values
    else:
        df['has_order'] = 0
        df['order_value'] = 0
    
    logger.info(f"Added derived columns. Final shape: {df.shape}")
    
    return df
