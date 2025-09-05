#!/usr/bin/env python3
"""Analyze user agent distribution in the processed data."""

import pandas as pd
import sys
from pathlib import Path

def analyze_user_agents(date: str = "2025-09-05"):
    """Analyze user agent distribution in the processed data."""
    
    # Load the processed data
    data_path = f"data/raw/{date}.parquet"
    if not Path(data_path).exists():
        print(f"Error: Processed data not found: {data_path}")
        return
    
    print(f"Analyzing user agents for date: {date}")
    print("=" * 50)
    
    # Load data
    df = pd.read_parquet(data_path)
    
    print(f"Total records: {len(df):,}")
    print()
    
    # User agent type distribution
    if 'user_agent_type' in df.columns:
        print("User Agent Type Distribution:")
        print("-" * 30)
        ua_type_counts = df['user_agent_type'].value_counts()
        for ua_type, count in ua_type_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{ua_type:15} {count:6,} ({percentage:5.1f}%)")
        print()
    
    # Browser model distribution
    if 'user_agent' in df.columns:
        print("Browser Model Distribution:")
        print("-" * 30)
        browser_counts = df['user_agent'].value_counts()
        for browser, count in browser_counts.head(10).items():
            percentage = (count / len(df)) * 100
            print(f"{browser:15} {count:6,} ({percentage:5.1f}%)")
        print()
    
    # Device type distribution
    if 'device_type' in df.columns:
        print("Device Type Distribution:")
        print("-" * 30)
        device_counts = df['device_type'].value_counts()
        for device, count in device_counts.head(10).items():
            percentage = (count / len(df)) * 100
            print(f"{device:20} {count:6,} ({percentage:5.1f}%)")
        print()
    
    # OS type distribution
    if 'os_type' in df.columns:
        print("OS Type Distribution:")
        print("-" * 30)
        os_counts = df['os_type'].value_counts()
        for os_type, count in os_counts.head(10).items():
            percentage = (count / len(df)) * 100
            print(f"{os_type:15} {count:6,} ({percentage:5.1f}%)")
        print()
    
    # Bot detection analysis
    if 'user_agent_type' in df.columns:
        bot_records = df[df['user_agent_type'] == 'Bot']
        app_records = df[df['user_agent_type'] == 'Application']
        
        print("Bot Detection Analysis:")
        print("-" * 30)
        print(f"Bot user agents:     {len(bot_records):6,} ({(len(bot_records)/len(df)*100):5.1f}%)")
        print(f"Application UAs:     {len(app_records):6,} ({(len(app_records)/len(df)*100):5.1f}%)")
        print(f"Web browsers:        {len(df[df['user_agent_type'] == 'Web Browser']):6,} ({(len(df[df['user_agent_type'] == 'Web Browser'])/len(df)*100):5.1f}%)")
        print()
    
    # Mobile vs Desktop analysis
    if 'device_type' in df.columns:
        mobile_devices = df[df['device_type'].str.contains('Phone|Tablet', case=False, na=False)]
        desktop_devices = df[df['device_type'].str.contains('PC|Desktop', case=False, na=False)]
        
        print("Mobile vs Desktop Analysis:")
        print("-" * 30)
        print(f"Mobile devices:      {len(mobile_devices):6,} ({(len(mobile_devices)/len(df)*100):5.1f}%)")
        print(f"Desktop devices:     {len(desktop_devices):6,} ({(len(desktop_devices)/len(df)*100):5.1f}%)")
        print()
    
    # Top user agent IDs
    if 'user_agent_id' in df.columns:
        print("Top 10 User Agent IDs:")
        print("-" * 30)
        ua_id_counts = df['user_agent_id'].value_counts().head(10)
        for ua_id, count in ua_id_counts.items():
            if pd.notna(ua_id):
                browser = df[df['user_agent_id'] == ua_id]['user_agent'].iloc[0]
                percentage = (count / len(df)) * 100
                print(f"ID {ua_id:4} ({browser:15}) {count:6,} ({percentage:5.1f}%)")

def main():
    if len(sys.argv) > 1:
        date = sys.argv[1]
    else:
        date = "2025-09-05"
    
    analyze_user_agents(date)

if __name__ == "__main__":
    main()
