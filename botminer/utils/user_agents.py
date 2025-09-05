"""User agent enrichment utilities."""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Global cache for user agent mappings
_user_agent_cache: Optional[Dict[int, Dict[str, str]]] = None


def load_user_agent_mappings(excel_path: str = "User Agent Major Details List.xlsx") -> Dict[int, Dict[str, str]]:
    """
    Load user agent ID mappings from Excel file.
    
    Args:
        excel_path: Path to the Excel file containing user agent mappings
        
    Returns:
        Dictionary mapping user agent ID to browser information
    """
    global _user_agent_cache
    
    if _user_agent_cache is not None:
        return _user_agent_cache
    
    logger.info(f"Loading user agent mappings from: {excel_path}")
    
    try:
        # Read the Excel file
        df = pd.read_excel(excel_path, sheet_name='Sheet1 (1)')
        
        # Create mapping dictionary
        user_agent_mappings = {}
        
        for _, row in df.iterrows():
            user_agent_id = int(row['id'])
            user_agent_mappings[user_agent_id] = {
                'name': str(row['name']),
                'userAgentType': str(row['userAgentType']),
                'browserModel': str(row['browserModel']),
                'majorVersion': int(row['majorVersion']) if pd.notna(row['majorVersion']) else None,
                'osType': str(row['osType']),
                'device': str(row['device'])
            }
        
        _user_agent_cache = user_agent_mappings
        logger.info(f"Loaded {len(user_agent_mappings)} user agent mappings")
        
        return user_agent_mappings
        
    except Exception as e:
        logger.error(f"Error loading user agent mappings: {e}")
        return {}


def enrich_user_agent(user_agent_id: int, excel_path: str = "User Agent Major Details List.xlsx") -> Dict[str, str]:
    """
    Enrich user agent ID with browser information.
    
    Args:
        user_agent_id: The 4-digit user agent ID
        excel_path: Path to the Excel file containing user agent mappings
        
    Returns:
        Dictionary with enriched user agent information
    """
    mappings = load_user_agent_mappings(excel_path)
    
    if user_agent_id in mappings:
        return mappings[user_agent_id]
    else:
        return {
            'name': f'Unknown User Agent {user_agent_id}',
            'userAgentType': 'Unknown',
            'browserModel': 'Unknown',
            'majorVersion': None,
            'osType': 'Unknown',
            'device': 'Unknown'
        }


def get_browser_type(user_agent_id: int, excel_path: str = "User Agent Major Details List.xlsx") -> str:
    """
    Get browser type for a user agent ID.
    
    Args:
        user_agent_id: The 4-digit user agent ID
        excel_path: Path to the Excel file containing user agent mappings
        
    Returns:
        Browser type string
    """
    enriched = enrich_user_agent(user_agent_id, excel_path)
    return enriched.get('browserModel', 'Unknown')


def get_user_agent_type(user_agent_id: int, excel_path: str = "User Agent Major Details List.xlsx") -> str:
    """
    Get user agent type (Web Browser, Bot, Application, etc.) for a user agent ID.
    
    Args:
        user_agent_id: The 4-digit user agent ID
        excel_path: Path to the Excel file containing user agent mappings
        
    Returns:
        User agent type string
    """
    enriched = enrich_user_agent(user_agent_id, excel_path)
    return enriched.get('userAgentType', 'Unknown')


def get_device_type(user_agent_id: int, excel_path: str = "User Agent Major Details List.xlsx") -> str:
    """
    Get device type for a user agent ID.
    
    Args:
        user_agent_id: The 4-digit user agent ID
        excel_path: Path to the Excel file containing user agent mappings
        
    Returns:
        Device type string
    """
    enriched = enrich_user_agent(user_agent_id, excel_path)
    return enriched.get('device', 'Unknown')


def get_os_type(user_agent_id: int, excel_path: str = "User Agent Major Details List.xlsx") -> str:
    """
    Get OS type for a user agent ID.
    
    Args:
        user_agent_id: The 4-digit user agent ID
        excel_path: Path to the Excel file containing user agent mappings
        
    Returns:
        OS type string
    """
    enriched = enrich_user_agent(user_agent_id, excel_path)
    return enriched.get('osType', 'Unknown')


def analyze_user_agent_distribution(user_agent_ids: pd.Series, excel_path: str = "User Agent Major Details List.xlsx") -> pd.DataFrame:
    """
    Analyze the distribution of user agent types in the data.
    
    Args:
        user_agent_ids: Series of user agent IDs
        excel_path: Path to the Excel file containing user agent mappings
        
    Returns:
        DataFrame with user agent analysis
    """
    mappings = load_user_agent_mappings(excel_path)
    
    # Create analysis dataframe
    analysis_data = []
    
    for ua_id in user_agent_ids.unique():
        if pd.notna(ua_id):
            enriched = enrich_user_agent(int(ua_id), excel_path)
            count = (user_agent_ids == ua_id).sum()
            
            analysis_data.append({
                'user_agent_id': ua_id,
                'count': count,
                'browser_model': enriched['browserModel'],
                'user_agent_type': enriched['userAgentType'],
                'device': enriched['device'],
                'os_type': enriched['osType']
            })
    
    return pd.DataFrame(analysis_data).sort_values('count', ascending=False)
